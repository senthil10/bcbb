import os
import unittest
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import tostring

from bcbio.utils import merge_flowcell_demux_summary


class DemuxFilesMerging(unittest.TestCase):
    """Test all functions to merge Demultiplexing result files.
    """

    def setUp(self):
        self.fc_dir = os.path.join(os.path.dirname(__file__), "data",
                                   "110221_empty_FC12345AAXX")
        self.expected_flowcell_demux_summary = ET.parse(
            os.path.join(self.fc_dir, 'Unaligned', 'Basecall_Stats_FC12345AAXX',
                         'Flowcell_demux_summary.xml'))

    def _compare_flowcell_demux_summary(self, fc1, fc2):
        """Compare the elements in two Flowcell_demux_summary.xml files.

        The children can be in any order, they should have exactly 8 lanes in
        total.
        """
        #The expected file is already sorted by lane, but we have to sort the
        #merged one in order to compare them
        r_expected = fc1.getroot()
        r_merged = fc2.getroot()
        #And we have to sort by attribute of the ElementTree
        r_merged.getchildren().sort(key=lambda x: x.attrib)
        return tostring(r_expected) == tostring(r_merged)

    def test_1_test_merge_flowcell_demux_stats(self):
        """Test merging Flowcell_demux_summary.xml files.
        """
        merged = merge_flowcell_demux_summary(self.fc_dir)
        self.assertTrue(self._compare_flowcell_demux_summary(
            self.expected_flowcell_demux_summary, merged))
