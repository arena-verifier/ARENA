\ Model example_solver
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.4545454545454545 var1 + 18.45454545454545 var2 + 1.909090909090909 var3
   + 5.454545454545454 var4
Subject To
 default_constraint: var1 + var2 + var3 + var4 = 1
 c1: - 0.9090909090909091 var1 + 5.090909090909091 var2
   + 3.090909090909091 var3 - 1.909090909090909 var4 = 0
Bounds
 var1 >= 0.0001
 var2 >= 0.0001
 var3 >= 0.0001
 var4 >= 0.0001
End
