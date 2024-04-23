import sys

import great_expectations as gx

context = gx.get_context()

print(context)

result = context.run_checkpoint(checkpoint_name="test_checkpoint")

if not result["success"]:
    print("Validation failed!")
    sys.exit(1)

print("Validation succeeded!")
sys.exit(0)