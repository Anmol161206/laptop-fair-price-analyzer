"""Tests for Find My Laptop: requirement extraction, spec profiles, filtering and explainable scoring."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.catalogue import load_catalogue
from src.recommender import (EXAMPLE_QUERIES, Requirements, extract_requirements, gpu_class_label, parse_budget,
                             recommend, spec_profile, weights_for, workload_fit)

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
        res = recommend(extract_requirements('college student, notes and browsing, budget 30k'))
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
        target = next(l for l in LAPTOPS if l.id in base)          # a laptop that is actually ranked for this need
        dearer = [replace(l, price_inr=l.price_inr * 1.5) if l.id == target.id else l for l in LAPTOPS]
        after = {r.laptop.id: r.score for r in recommend(req, laptops=dearer, top_n=50).recommendations}
        self.assertLessEqual(after[target.id], base[target.id])

    def test_portability_priority_favours_lighter(self):
        heavy_first = recommend(Requirements(workloads=['everyday'], portability='Low'), top_n=50).recommendations
        light_first = recommend(Requirements(workloads=['everyday'], portability='High'), top_n=50).recommendations
        known = lambda recs: [r.laptop.weight_kg for r in recs if r.laptop.weight_kg is not None][:5]
        avg = lambda recs: sum(known(recs)) / len(known(recs))
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


class TestExpandedCatalogue(unittest.TestCase):
    PERSONAS = {
        'college': ('College student, mostly notes, browsing and online classes. Budget ₹40,000.', 40_000),
        'engineering': ('First-year engineering student, MATLAB and AutoCAD for assignments. Budget 55k.', 55_000),
        'cs': ('Computer science student, coding in Python and Java with VS Code. Budget ₹75,000.', 75_000),
        'ml': ('Student learning machine learning and deep learning with PyTorch. Budget 90k.', 90_000),
        'casual_gamer': ('I want to play Valorant and Minecraft casually. Budget ₹65,000.', 65_000),
        'gaming': ('Gaming enthusiast, I play GTA V and Cyberpunk. Budget 1.2 lakh.', 120_000),
        'cad': ('Mechanical engineering student using SolidWorks and ANSYS. Budget under 1 lakh.', 100_000),
        'designer': ('Graphic designer working in Photoshop, Illustrator and Figma. Budget 1,10,000.', 110_000),
        'business': ('Business professional, Excel, PowerPoint and video meetings, I travel a lot. Budget 85k.', 85_000),
        'premium': ('I want a premium thin and light ultrabook with long battery life for work. Budget ₹1,60,000.', 160_000),
    }

    def run_persona(self, key):
        text, budget = self.PERSONAS[key]
        req = extract_requirements(text)
        self.assertEqual(req.budget_max, budget)
        return req, recommend(req)

    def test_all_personas_respect_budget_and_os(self):
        for key in self.PERSONAS:
            req, res = self.run_persona(key)
            with self.subTest(key):
                self.assertTrue(res.recommendations)
                if res.status != 'over_budget':
                    self.assertTrue(all(r.laptop.price_inr is not None and r.laptop.price_inr <= req.budget_max
                                        for r in res.recommendations))
                if res.profile.required_os:
                    self.assertTrue(all(r.laptop.os == res.profile.required_os for r in res.recommendations))

    def test_persona_specific_expectations(self):
        _, gaming = self.run_persona('gaming')
        self.assertEqual(gaming.status, 'ok')
        self.assertTrue(all(r.laptop.gpu_dedicated and r.laptop.gpu_tier >= 3 for r in gaming.recommendations))
        _, cad = self.run_persona('cad')
        self.assertTrue(all(r.laptop.os == 'Windows' and r.laptop.gpu_tier >= 2 for r in cad.recommendations))
        _, cs = self.run_persona('cs')
        self.assertEqual(cs.status, 'ok')
        self.assertTrue(all(r.laptop.ram_gb >= 16 for r in cs.recommendations))
        _, casual = self.run_persona('casual_gamer')       # no dedicated GPU exists under 65k in the catalogue
        self.assertEqual(casual.status, 'below_minimum')
        self.assertTrue(all(r.gaps for r in casual.recommendations))
        _, college = self.run_persona('college')
        self.assertEqual(college.status, 'ok')
        _, premium = self.run_persona('premium')
        self.assertTrue(all(r.laptop.weight_kg is not None and r.laptop.weight_kg <= 1.5 for r in premium.recommendations))

    def test_unverified_price_never_fits_a_budget(self):
        target = next(l for l in LAPTOPS if l.price_inr and l.price_inr < 70_000)
        unknown = [replace(l, price_inr=None) if l.id == target.id else l for l in LAPTOPS]
        res = recommend(Requirements(workloads=['everyday'], budget_max=500_000), laptops=unknown, top_n=100)
        self.assertNotIn(target.id, [r.laptop.id for r in res.recommendations])
        self.assertEqual(res.excluded[target.id], 'price not verified')
        anyprice = recommend(Requirements(workloads=['everyday']), laptops=unknown, top_n=100)   # no budget: listed, no value credit
        rec = next(r for r in anyprice.recommendations if r.laptop.id == target.id)
        self.assertEqual(next(c['score'] for c in rec.breakdown if c['key'] == 'value'), 0.0)
        self.assertIsNone(rec.budget_gap)

    def test_unknown_weight_is_neutral_and_flagged(self):
        lap = next(l for l in LAPTOPS if l.weight_kg is None)
        from src.recommender import portability_score
        self.assertEqual(portability_score(lap), 0.5)
        res = recommend(Requirements(workloads=['everyday'], portability='High'), laptops=(lap,))
        self.assertTrue(any('Weight is not stated' in c for c in res.recommendations[0].cons))

    def test_top_three_are_varied(self):
        from src.recommender import _series
        for key in ('cs', 'business', 'designer', 'gaming'):
            _, res = self.run_persona(key)
            series = [_series(r.laptop) for r in res.recommendations]
            with self.subTest(key):
                self.assertEqual(len(series), len(set(series)))


class TestGpuClassification(unittest.TestCase):
    """Regression: Apple silicon GPUs are integrated SoC graphics; the tier is a performance class only."""

    def test_apple_silicon_is_integrated_without_cuda(self):
        macs = [l for l in LAPTOPS if l.brand == 'Apple']
        self.assertTrue(macs)
        for lap in macs:
            with self.subTest(lap.id):
                self.assertFalse(lap.gpu_dedicated)
                self.assertFalse(lap.cuda)
                self.assertIn('Integrated', gpu_class_label(lap).title())
                self.assertNotIn('dedicated', gpu_class_label(lap).lower())

    def test_m5_pro_is_pro_class_integrated_not_dedicated(self):
        from src.ui_components import comparison_table
        mbp = next(l for l in LAPTOPS if 'M5 Pro' in l.gpu)
        self.assertEqual(mbp.gpu_tier, 3)
        self.assertEqual(gpu_class_label(mbp), 'Integrated, Pro-class performance')
        html = comparison_table([mbp, next(l for l in LAPTOPS if l.brand == 'Apple' and l is not mbp)])
        self.assertNotIn('Dedicated', html)

    def test_dedicated_and_cuda_only_for_nvidia(self):
        for lap in LAPTOPS:
            with self.subTest(lap.id):
                self.assertEqual(lap.cuda, 'NVIDIA' in lap.gpu.upper())
                if lap.gpu_dedicated:
                    self.assertIn('dedicated', gpu_class_label(lap).lower())
                else:
                    self.assertNotIn('dedicated', gpu_class_label(lap).lower())

    def test_cuda_workloads_flag_macs(self):
        result = recommend(extract_requirements('Deep learning with PyTorch, budget 3 lakh, I prefer a Mac'))
        macs = [r for r in result.recommendations if r.laptop.os == 'macOS']
        self.assertTrue(macs)
        for rec in macs:
            self.assertTrue(any('CUDA' in c for c in rec.cons))
            self.assertLess(workload_fit(rec.laptop, result.profile), 1.0)

    def test_gaming_does_not_recommend_macs(self):
        for q in ('I play Valorant, budget 3.5 lakh, I prefer a Mac', 'AAA gaming laptop, budget 3.5 lakh, prefer macOS'):
            with self.subTest(q):
                result = recommend(extract_requirements(q))
                self.assertTrue(result.recommendations)
                self.assertTrue(all(r.laptop.os == 'Windows' for r in result.recommendations))


if __name__ == '__main__':
    unittest.main()
