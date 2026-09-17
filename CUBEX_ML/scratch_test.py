import pandas as pd
from carbon_capture.pipeline.end_to_end import EndToEndPipeline

pipeline = EndToEndPipeline()
df = pd.read_csv("data/synthetic/test_corrupted_mass_balance.csv")
res = pipeline.run(df)
decision = res["decision_engine_output"]
print("Decision:", decision.decision)
print("F_valid:", decision.F_valid)
print("Max norm res:", res["epinn_output"].max_normalized_residual)
print("Mean norm res:", res["epinn_output"].mean_normalized_residual)
if decision.decision == "VERIFIED":
    print("Why verified? Rejection reasons:", decision.rejection_reasons)
