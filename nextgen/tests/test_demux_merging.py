import os
import unittest
from xml.etree import ElementTree as ET

from bcbio.utils import merge_flowcell_demux_summary

class DemuxFilesMerging(unittest.TestCase):
	"""Test all functions to merge Demultiplexing result files.
	"""

	def setUp(self):
		self.fc_dir = os.path.join(os.path.dirname(__file__), "data", "110221_empty_FC12345AAXX")
		self.expected_flowcell_demux_summary = ET.Element('Summary')
		for i in range(1,9):
			self.expected_flowcell_demux_summary.append(ET.Element('Lane', index=str(i)))


	def _compare_flowcell_demux_summary(fc1, fc2):
		"""Compare the elements in two Flowcell_demux_summary.xml files.

		The children can be in any order, and they should have exactly 8 lanes in
		total.
		"""
		


	def test_1_test_merge_flowcell_demux_stats(self):
		"""Test merging Flowcell_demux_summary.xml files.
		"""

