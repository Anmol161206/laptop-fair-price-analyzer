"""Tests for Find My Laptop: requirement extraction, spec profiles, filtering and explainable scoring."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.catalogue import load_catalogue
from src.recommender import (EXAMPLE_QUERIES, Requirements, extract_requirements, parse_budget, recommend,
                             spec_profile, weights_for, workload_fit)

LAPTOPS = load_catalogue()[1]


class TestBudgetParsing(unittest.TestCase):
    def test_formats(self):
        cases = {
            'budget ₹70,000': 70_000, 'Rs 85000 max': 85_000, 'under 90k': 90_000, 'up to 1.2 lakh': 120_000,
            'between 70k and 90k': 90_000, '60-80k please': 80_000, 'within 1 lakh': 100_000, 'budget 150000': 150_000,
            'around 80k': 88_000,
        }
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertEqual(parse_budget(text), expected)

    def test_common_indian_phrasings(self):
        for text in ['₹80,000', '80k', '80K', '80 thousand', 'Rs. 80,000/-', 'INR 80000', '0.8 lakh', '80,000',
                     'budget is 80,000 rupees']:
            with self.subTest(text):
                self.assertEqual(parse_budget(text), 80_000)
        for text in ['under 1 lakh', 'max 1L', '1,00,000', '₹1,00,000', 'within 1 lac']:
            with self.subTest(text):
                self.assertEqual(parse_budget(text), 100_000)

    def test_ignores_non_money_numbers(self):
        for text in ['I edit 4k video', 'need a 4k display', '16 GB RAM and 512 GB SSD', 'play at 144 hz', '']:
            with self.subTest(text):
                self.assertIsNone(parse_budget(text))


class TestExtraction(unittest.TestCase):
    def test_workloads_from_degree_software_and_games(self):
        cases = {
            'B.Tech computer science, I code in Python': {'programming'},
            'I want to learn machine learning with PyTorch': {'data_science'},
            'Mechanical student using SolidWorks and AutoCAD': {'engineering'},
            'I edit YouTube videos in Premiere Pro': {'video'},
            'Graphic designer, Photoshop and Figma': {'design'},
            'Blender and Unity for game dev': {'3d'},
            'I play Valorant and Minecraft': {'gaming_esports'},
            'I want to play GTA V and Cyberpunk': {'gaming_aaa'},
            'MBA student, mostly Excel and PowerPoint': {'office'},
        }
        for text, expected in cases.items():
            with self.subTest(text):
                self.assertTrue(expected <= set(extract_requirements(text).workloads))

    def test_os_rules(self):
        self.assertEqual(extract_requirements('I need SolidWorks').required_os, 'Windows')
        self.assertEqual(extract_requirements('I edit in Final Cut Pro').required_os, 'macOS')
        self.assertEqual(extract_requirements('I prefer a MacBook').os_preference, 'macOS')
        self.assertEqual(extract_requirements('no mac please').os_preference, 'Windows')
        # 'machine learning' must not be read as 'Mac'
        self.assertEqual(extract_requirements('machine learning student').os_preference, 'Any')

    def test_preferences(self):
        r = extract_requirements('light laptop with long battery life, I travel a lot')
        self.assertEqual((r.portability, r.battery), ('High', 'High'))
        self.assertEqual(extract_requirements("weight doesn't matter, desktop replacement").portability, 'Low')

    def test_ambiguous_input_handled(self):
        r = extract_requirements('I need a laptop')
        self.assertEqual(r.workloads, ['everyday'])
        self.assertIsNone(r.budget_max)
        self.assertEqual(len(r.notes), 2)          # no workload recognised + no budget
        self.assertEqual(extract_requirements('').workloads, ['everyday'])

    def test_conflict_note(self):
        r = extract_requirements('I want a MacBook for SolidWorks')
        self.assertEqual(r.required_os, 'Windows')
        self.assertTrue(any('needs Windows' in n for n in r.notes))


class TestSpecProfile(unittest.TestCase):
    def test_combines_most_demanding(self):
        p = spec_profile(Requirements(workloads=['office', 'gaming_aaa']))
        self.assertEqual(p.rec_gpu, 4)
        self.assertEqual(p.required_os, 'Windows')
        self.assertEqual(p.prefer_refresh, 144)
        self.assertTrue(spec_profile(Requirements(workloads=['data_science'])).prefer_cuda)


class TestRecommend(unittest.TestCase):
    def test_never_exceeds_budget_when_matches_exist(self):
        for budget in (100_000, 130_000, 160_000, 250_000):
            for workloads in (['everyday'], ['programming'], ['gaming_esports'], ['engineering']):
                res = recommend(Requirements(workloads=workloads, budget_max=budget))
                with self.subTest(budget=budget, workloads=workloads):
                    if res.status == 'ok':
                        self.assertTrue(all(r.laptop.price_inr <= budget for r in res.recommendations))
                        self.assertTrue(all(not r.gaps for r in res.recommendations))

    def test_os_requirement_is_hard(self):
        for workloads, os_name in ((['gaming_aaa'], 'Windows'), (['office'], 'macOS')):
            req = Requirements(workloads=workloads, os_preference='Any' if os_name == 'Windows' else 'macOS')
            self.assertTrue(all(r.laptop.os == os_name for r in recommend(req).recommendations))

    def test_at_most_three_and_deterministic(self):
        req = Requirements(workloads=['programming'], budget_max=200_000)
        a, b = recommend(req), recommend(req)
        self.assertLessEqual(len(a.recommendations), 3)
        self.assertEqual([r.laptop.id for r in a.recommendations], [r.laptop.id for r in b.recommendations])
        scores = [r.score for r in a.recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_fallbacks(self):
        low = recommend(Requirements(workloads=['3d'], budget_max=80_000))
        self.assertIn(low.status, ('below_minimum', 'over_budget'))
        self.assertTrue(low.recommendations)
        none_left = recommend(Requirements(workloads=['everyday'], budget_max=20_000))
        self.assertEqual(none_left.status, 'over_budget')
        self.assertTrue(all(r.budget_gap < 0 for r in none_left.recommendations))

    def test_hard_budget_never_exceeded_in_main_results(self):
        texts = ['student, budget ₹80,000', 'coding laptop 80k', 'gaming under 1 lakh', 'office work, max 1.5L',
                 'video editing within 2 lakh', 'CAD student, budget 1,30,000']
        for text in texts:
            req = extract_requirements(text)
            res = recommend(req)
            with self.subTest(text):
                if res.status in ('ok', 'below_minimum'):
                    self.assertTrue(all(r.laptop.price_inr <= req.budget_max for r in res.recommendations))
                else:
                    self.assertTrue(all(r.budget_gap < 0 for r in res.recommendations))   # flagged as over budget

    def test_contradictory_os_preferences(self):
        mac_gamer = recommend(Requirements(workloads=['gaming_esports'], os_preference='macOS'))
        self.assertTrue(all(r.laptop.os == 'Windows' for r in mac_gamer.recommendations))
        self.assertIn('needs Windows', mac_gamer.message)
        mac_cad = extract_requirements('I want a MacBook but need SolidWorks')
        self.assertEqual(mac_cad.required_os, 'Windows')
        self.assertTrue(all(r.laptop.os == 'Windows' for r in recommend(mac_cad).recommendations))
        final_cut = recommend(extract_requirements('I edit in Final Cut Pro, prefer Windows'))
        self.assertTrue(all(r.laptop.os == 'macOS' for r in final_cut.recommendations))

    def test_low_budget_is_not_relaxed(self):
        res = recommend(extract_requirements('college student, notes and browsing, budget 45k'))
        self.assertEqual(res.status, 'over_budget')
        self.assertIn('No verified laptop', res.message)
        self.assertTrue(res.profile.rec_ram >= 8)               # spec advice still produced

    def test_more_ram_never_lowers_fit(self):
        p = spec_profile(Requirements(workloads=['data_science']))
        for lap in LAPTOPS:
            self.assertGreaterEqual(workload_fit(replace(lap, ram_gb=lap.ram_gb * 2), p), workload_fit(lap, p))

    def test_higher_price_never_raises_score(self):
        req = Requirements(workloads=['programming'])
        base = {r.laptop.id: r.score for r in recommend(req, top_n=50).recommendations}
        target = LAPTOPS[3]
        dearer = [replace(l, price_inr=l.price_inr * 1.5) if l.id == target.id else l for l in LAPTOPS]
        after = {r.laptop.id: r.score for r in recommend(req, laptops=dearer, top_n=50).recommendations}
        self.assertLessEqual(after[target.id], base[target.id])

    def test_portability_priority_favours_lighter(self):
        heavy_first = recommend(Requirements(workloads=['everyday'], portability='Low'), top_n=50).recommendations
        light_first = recommend(Requirements(workloads=['everyday'], portability='High'), top_n=50).recommendations
        avg = lambda recs: sum(r.laptop.weight_kg for r in recs[:3]) / 3
        self.assertLessEqual(avg(light_first), avg(heavy_first))

    def test_weights_sum_to_one(self):
        for port in ('Low', 'Medium', 'High'):
            for screen in ('Any', 'Compact (14" or smaller)'):
                w = weights_for(Requirements(portability=port, screen=screen, budget_max=100_000))
                self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_explanations_reference_real_specs(self):
        for name, text in EXAMPLE_QUERIES.items():
            for r in recommend(extract_requirements(text)).recommendations:
                with self.subTest(name=name, laptop=r.laptop.id):
                    self.assertTrue(r.pros or r.cons)
                    for line in r.pros + r.cons:
                        if 'GB RAM' in line:
                            self.assertIn(f'{r.laptop.ram_gb} GB RAM', line)
                        if ' kg' in line:
                            self.assertIn(f'{r.laptop.weight_kg:g} kg', line)
                    self.assertTrue(0 <= r.score <= 1)
                    self.assertAlmostEqual(sum(c['weight'] for c in r.breakdown), 1.0)

    def test_persona_scenarios(self):
        def run(text):
            return recommend(extract_requirements(text)).recommendations
        gamer = run(EXAMPLE_QUERIES['Gamer'])
        self.assertTrue(all(r.laptop.gpu_dedicated and r.laptop.os == 'Windows' for r in gamer))
        engineer = run(EXAMPLE_QUERIES['Mechanical engineer'])
        self.assertTrue(all(r.laptop.os == 'Windows' and r.laptop.price_inr <= 160_000 for r in engineer))
        business = run(EXAMPLE_QUERIES['Business user'])
        self.assertTrue(all(r.laptop.os == 'macOS' for r in business))
        ml = run('data science student, deep learning with PyTorch, budget 1.6 lakh')
        self.assertTrue(ml[0].laptop.cuda)
        budget_student = run('first year BA student, notes, browsing and Netflix, budget 65k')
        self.assertTrue(budget_student and all(r.laptop.price_inr <= 65_000 for r in budget_student))
        designer = run('graphic designer using Photoshop and Figma, light laptop, under 1 lakh')
        self.assertTrue(all(r.laptop.ram_gb >= 16 and r.laptop.price_inr <= 100_000 for r in designer))
        programmer = run('software developer, VS Code and Docker, budget 90k')
        self.assertTrue(all(r.laptop.ram_gb >= 16 for r in programmer))


if __name__ == '__main__':
    unittest.main()
