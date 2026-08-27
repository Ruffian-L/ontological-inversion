import unittest
from memory_support.protocol import records
from memory_support.selector import select_plateau
from memory_support.analyze import aggregate
from memory_support.judge import opaque

class FrozenSelectorTest(unittest.TestCase):
    def test_corpus_is_cluster_split_and_pregenerated(self):
        rows=records(); self.assertEqual(len(rows),24*4*2)
        self.assertEqual(len({r["cluster_id"] for r in rows if r["split"]=="calibration"}),12)
        self.assertEqual(len({r["cluster_id"] for r in rows if r["split"]=="heldout"}),12)
        self.assertTrue(all("evaluat" in r["prompt"].lower() for r in rows))
    def test_worst_margin_plateau_midpoint(self):
        gains=[0,.02,.04,.06,.08,.10,.12,.14,.16]; th={"S":.1,"F":.5,"U":.5,"H":.2,"D":.15,"V":.55}
        cells=[]
        for g in gains:
            passing=.04<=g<=.10
            cells.append({"gain":g,"S_lcb":.2 if passing else 0,"F_lcb":.7,"U_lcb":.7,"H_ucb":.1,"D_ucb":.05,"V_lcb":.7})
        got=select_plateau(cells,gains,th); self.assertEqual(got["status"],"selected"); self.assertAlmostEqual(got["gain"],.07)
    def test_abstains_without_three_points(self):
        gains=[0,.02,.04]; th={"S":.1,"F":.5,"U":.5,"H":.2,"D":.15,"V":.55}
        cells=[{"gain":g,"S_lcb":.2 if g<.04 else 0,"F_lcb":.7,"U_lcb":.7,"H_ucb":.1,"D_ucb":.05,"V_lcb":.7} for g in gains]
        self.assertEqual(select_plateau(cells,gains,th)["status"],"abstain")
    def test_specificity_uses_strongest_control(self):
        cfg={"gains":[.08]}; rows=[]
        for arm,val in (("matched",1.0),("wrong_memory",.7),("random",.1),("blank",0.0)):
            for seed in range(3):
                rows.append({"cluster_id":"x","gain":.08,"arm":arm,"family":"entailed","form_id":0,"F":val,"U":val,"H":0,"D":0})
        cell=aggregate(rows,cfg)[0]
        self.assertLessEqual(cell["S_lcb"],.3000001)
    def test_judge_identifier_is_opaque(self):
        self.assertNotIn("matched",opaque("matched-gain-.08-seed-1"))
        self.assertEqual(len(opaque("x")),24)
if __name__=="__main__": unittest.main()
