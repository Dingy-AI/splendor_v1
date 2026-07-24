
from splendor_v1.env.core.cost_lookup_table_v3 import T1_PAYMENT_LOOKUP, T2_PAYMENT_LOOKUP, T3_PAYMENT_LOOKUP


print(set(T1_PAYMENT_LOOKUP).issubset(set(T3_PAYMENT_LOOKUP)))
print(set(T2_PAYMENT_LOOKUP).issubset(set(T3_PAYMENT_LOOKUP)))

#True
#True