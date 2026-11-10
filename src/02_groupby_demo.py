"""
A tiny, deliberate demonstration of the groupby technique used in section 7
of notebook 01 -- with a table small enough to hold in your head.
"""
import pandas as pd

# A pretend table: 6 hospital stays.
#   age_bracket : which decade the patient is in
#   readmit_30d : 1 = came back within 30 days, 0 = did not
toy = pd.DataFrame({
    "patient":      ["A", "B", "C", "D", "E", "F"],
    "age_bracket":  ["[20-30)", "[20-30)", "[20-30)", "[70-80)", "[70-80)", "[70-80)"],
    "readmit_30d":  [1, 1, 0, 1, 0, 0],
})

print("The toy table:")
print(toy.to_string(index=False))

# Group by age bracket and average the 0/1 column.
result = toy.groupby("age_bracket")["readmit_30d"].mean()
print()
print("groupby result (the average of the 0/1 column per group):")
print(result.to_string())

print()
print("Interpretation:")
print("  2 of 3 young patients came back -> 2/3 = 0.667 (66.7%)")
print("  1 of 3 older patients came back -> 1/3 = 0.333 (33.3%)")
print("  Averaging 1s and 0s IS the same thing as computing a percentage.")

print()
print("The same thing with counts, using .agg():")
agg = (toy.groupby("age_bracket")["readmit_30d"]
          .agg(n="count", readmit_rate="mean").round(3))
print(agg.to_string())
print()
print("Rule of thumb: a percentage is meaningless without its n.")
print("0.667 from n=3 is a rumour; 0.667 from n=3,000 is evidence.")
