"""Headless app tests for the Find My Laptop and Compare tabs (Streamlit AppTest)."""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.catalogue import OFFICIAL_URL_PREFIXES
from src.recommender import EXAMPLE_QUERIES, extract_requirements, recommend
from src.ui_components import price_text

APP_PATH = str(PROJECT_ROOT / 'app.py')


class TestFindMyLaptopApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest
        cls.AppTest = AppTest

    def run_example(self, name):
        at = self.AppTest.from_file(APP_PATH, default_timeout=60).run()
        at.button(key=f'fm_ex_{name}').click().run()
        at.button(key='fm_find').click().run()
        self.assertEqual(len(at.exception), 0, [e.value for e in at.exception])
        return at

    def test_four_tabs_and_idle_state(self):
        at = self.AppTest.from_file(APP_PATH, default_timeout=60).run()
        self.assertEqual(len(at.tabs), 4)
        self.assertEqual(len(at.exception), 0)
        self.assertNotIn('fm_result', at.session_state)

    def test_results_match_engine(self):
        for name in EXAMPLE_QUERIES:
            with self.subTest(name):
                at = self.run_example(name)
                expected = recommend(extract_requirements(EXAMPLE_QUERIES[name]))
                shown = at.session_state['fm_result']
                self.assertEqual([r.laptop.id for r in shown.recommendations], [r.laptop.id for r in expected.recommendations])
                html = ' '.join(m.value for m in at.markdown)
                for r in shown.recommendations:
                    self.assertIn(price_text(r.laptop.price_inr), html)          # verified price shown
                    self.assertIn('verified', html)

    def test_store_links_are_official(self):
        at = self.run_example('Gamer')
        html = ' '.join(m.value for m in at.markdown)
        for r in at.session_state['fm_result'].recommendations:
            self.assertTrue(r.laptop.url.startswith(OFFICIAL_URL_PREFIXES))
            self.assertIn(r.laptop.url, html)

    def test_editing_requirements_updates_results(self):
        at = self.run_example('Mechanical engineer')
        before = [r.laptop.id for r in at.session_state['fm_result'].recommendations]
        at.number_input(key='fm_budget').set_value(100000).run()
        after = at.session_state['fm_result']
        self.assertEqual(len(at.exception), 0)
        self.assertNotEqual(before, [r.laptop.id for r in after.recommendations])
        self.assertNotEqual(after.status, 'ok')        # nothing meets CAD minimums under 1 lakh -> honest fallback

    def test_compare_preselects_recommendations(self):
        at = self.run_example('Business user')
        names = [r.laptop.display_name for r in at.session_state['fm_result'].recommendations]
        self.assertEqual(at.multiselect(key='cmp_select').value, names)

    def test_low_budget_shows_no_cards_and_keeps_specs(self):
        at = self.AppTest.from_file(APP_PATH, default_timeout=60).run()
        at.text_area(key='fm_query').set_value('college student, notes and browsing, budget 45k').run()
        at.button(key='fm_find').click().run()
        self.assertEqual(len(at.exception), 0)
        html = ' '.join(m.value for m in at.markdown)
        self.assertEqual(at.session_state['fm_result'].status, 'over_budget')
        self.assertNotIn('class="lap', html)                       # no recommendation cards
        self.assertIn('Recommended specifications', html)
        self.assertTrue(any('above your budget' in e.label for e in at.expander))

    def test_contradictory_os_note_shown(self):
        at = self.run_example('Gamer')
        at.selectbox(key='fm_os').set_value('macOS').run()
        html = ' '.join(m.value for m in at.markdown)
        self.assertIn('needs Windows', html)
        self.assertTrue(all(r.laptop.os == 'Windows' for r in at.session_state['fm_result'].recommendations))

    def test_historical_estimator_still_separate(self):
        at = self.run_example('CS student')
        html = ' '.join(m.value for m in at.markdown)
        self.assertIn('Historical reference model, not a current price', html)
        self.assertIn('Estimate Price', at.info[0].value)        # estimator untouched by the finder


if __name__ == '__main__':
    unittest.main()
