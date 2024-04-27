import sys
import os
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues, \
  TestNumberOfConstantColumns, TestNumberOfDuplicatedRows, TestNumberOfDuplicatedColumns, TestColumnsType, \
  TestNumberOfDriftedColumns

def main():
  tests = TestSuite(tests=[
    TestNumberOfColumnsWithMissingValues(),
    TestNumberOfRowsWithMissingValues(),
    TestNumberOfConstantColumns(),
    TestNumberOfDuplicatedRows(),
    TestNumberOfDuplicatedColumns(),
    TestColumnsType(),
    TestNumberOfDriftedColumns(),
    NoTargetPerformanceTestPreset(),
    DataStabilityTestPreset()
  ])

  current_path = os.path.join('data', 'validate', 'current_data.csv')
  reference_path = os.path.join('data', 'processed', 'reference_data.csv')

  current = pd.read_csv(current_path)
  reference = pd.read_csv(reference_path)

  tests.run(reference_data=reference, current_data=current)

  tests.save_html("reports/stability_tests.html")

  # test_results = tests.as_dict()

  # Check if any test failed
  # if test_results['summary']['failed_tests'] > 0:
  #   print("Some tests failed:")
  #   print(test_results['summary']['failed_tests'])
  #   sys.exit(1)
  # else:
  #   print("All tests passed!")