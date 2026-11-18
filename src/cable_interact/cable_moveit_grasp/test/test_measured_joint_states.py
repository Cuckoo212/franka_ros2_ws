import pathlib
import sys
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from joint_state_store import MeasuredJointStore


class StoreTests(unittest.TestCase):
    def update(self, store, source, value, stamp=10_000_000_000, received=10.):
        names=sorted(store.owners[source])
        store.update(source,names,[value]*len(names),[],[],stamp,received)

    def test_no_default_arm_before_measurement(self):
        store=MeasuredJointStore()
        self.update(store,'b',.4)
        self.assertIsNone(store.snapshot(10_000_000_000,10.))
        self.update(store,'a',.7)
        result=store.snapshot(10_000_000_000,10.)
        self.assertEqual(len(result[0]),14)
        for name,value in zip(result[0],result[1]):
            self.assertEqual(value,.7 if name.startswith('a_') else .4)

    def test_other_arm_zero_entries_cannot_overwrite(self):
        store=MeasuredJointStore()
        self.update(store,'a',.7)
        names=sorted(store.owners['b'])+sorted(store.owners['a'])
        ignored=store.update('b',names,[.4]*7+[0.]*7,[],[],10_000_000_000,10.)
        self.assertEqual(len(ignored),7)
        result=store.snapshot(10_000_000_000,10.)
        positions=dict(zip(result[0],result[1]))
        self.assertEqual(positions['a_fr3_joint1'],.7)

    def test_stale_arm_is_not_restamped(self):
        store=MeasuredJointStore()
        self.update(store,'a',.7)
        self.update(store,'b',.4,stamp=12_000_000_000,received=12.)
        self.assertIsNone(store.snapshot(12_000_000_000,12.))
        # An old sensor timestamp arriving now must not be treated as fresh.
        self.update(store,'a',.7,received=12.)
        self.assertIsNone(store.snapshot(12_000_000_000,12.))

    def test_out_of_order_data_cannot_rewind(self):
        store=MeasuredJointStore()
        self.update(store,'a',.7)
        self.update(store,'b',.4)
        self.update(store,'a',0.,stamp=9_000_000_000,received=10.1)
        result=store.snapshot(10_000_000_000,10.1)
        self.assertEqual(dict(zip(result[0],result[1]))['a_fr3_joint1'],.7)
        self.assertEqual(result[4],10_000_000_000)

    def test_malformed_arrays_rejected_without_partial_update(self):
        store=MeasuredJointStore()
        with self.assertRaises(ValueError):
            store.update('a',['a_fr3_joint1'],[],[],[],10_000_000_000,10.)
        self.assertEqual(store.values,{})


if __name__=='__main__':
    unittest.main()
