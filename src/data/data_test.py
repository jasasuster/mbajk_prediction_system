import sys
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset
from evidently.tests import TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues, \
  TestNumberOfConstantColumns, TestNumberOfDuplicatedRows, TestNumberOfDuplicatedColumns, TestColumnsType, \
  TestNumberOfDriftedColumns

if __name__ == "__main__":
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

  current = pd.read_csv("data/validate/current_data.csv")
  reference = pd.read_csv("data/processed/reference_data.csv")

  tests.run(reference_data=reference, current_data=current)

  tests.save_html("reports/stability_tests.html")

  test_results = tests.as_dict()

  # Check if any test failed
  if test_results['summary']['failed_tests'] > 0:
    print("Some tests failed:")
    print(test_results['summary']['failed_tests'])
    sys.exit(1)
  else:
    print("All tests passed!")