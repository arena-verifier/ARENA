#include <stdlib.h>
#include <stdio.h>
#include "gurobi_c.h"

int main(int argc,char *argv[])
{
  GRBenv   *env   = NULL;
  GRBmodel *model = NULL;
  int       error = 0;
  double    sol[4];
  int       ind[4];
  double    val[4];
  double    obj[4];
  double    lbs[4];
  double    ubs[4];
  char      vtype[4];
  int       optimstatus;
  double    objval;

  /* Create environment */
  error = GRBemptyenv(&env);
  if (error) goto QUIT;

  error = GRBsetstrparam(env, "LogFile", "example_solver.log");
  if (error) goto QUIT;

  error = GRBstartenv(env);
  if (error) goto QUIT;

  /* Create an empty model */
  error = GRBnewmodel(env, &model, "example_solver", 0, NULL, NULL, NULL, NULL, NULL);
  if (error) goto QUIT;

  /* Add variables, a1,a2,a3,a4, and create the objective function */
  obj[0] = 5.0/11; obj[1] = 203.0/11; obj[2] = 21.0/11; obj[3]= 60.0/11;
  vtype[0] = GRB_CONTINUOUS; vtype[1] = GRB_CONTINUOUS; vtype[2] = GRB_CONTINUOUS; vtype[3] = GRB_CONTINUOUS;
  lbs[0] = 0.0001; lbs[1] = 0.0001; lbs[2] = 0.0001; lbs[3] = 0.0001;
  ubs[0] = GRB_INFINITY; ubs[1] = GRB_INFINITY; ubs[2] = GRB_INFINITY; ubs[3] = GRB_INFINITY;
  error = GRBaddvars(model, 4, 0, NULL, NULL, NULL, obj, lbs, ubs, vtype,
                     NULL);
  if (error) goto QUIT;

  /* The normalization constraint that the sum of the weights must be 1 */

  ind[0] = 0; ind[1] = 1; ind[2] = 2; ind[3] = 3; 
  val[0] = 1; val[1] = 1; val[2] = 1; val[3] = 1;
  error = GRBaddconstr(model, 4, ind, val, GRB_EQUAL, 1.0, "default_constraint");
  if (error) goto QUIT;

  /* First constraint for variable a cancellation */
  /*
  ind[0] = 0; ind[1] = 1; ind[2] = 2; ind[3] = 3; 
  val[0] = -15.0/11; val[1] = 117.0/11; val[2] = -81.0/11; val[3] = -26.0/11;

  error = GRBaddconstr(model, 4, ind, val, GRB_EQUAL, 0.0, "c0");
  if (error) goto QUIT;
  */
  /* Second constraint for variable b cancellation */
  ind[0] = 0; ind[1] = 1; ind[2] = 2; ind[3] = 3; 
  val[0] = -10.0/11; val[1] = 56.0/11; val[2] = 34.0/11; val[3] = -21.0/11;

  error = GRBaddconstr(model, 4, ind, val, GRB_EQUAL, 0.0, "c1");
  if (error) goto QUIT;

  /* Third constraint for variable d cancellation */
  ind[0] = 0; ind[1] = 1; ind[2] = 2; ind[3] = 3; 
  val[0] = -25.0/11; val[1] = -289.0/11; val[2] = -3.0/11; val[3] = 52.0/11;

  error = GRBaddconstr(model, 4, ind, val, GRB_EQUAL, 0.0, "c2");
  if (error) goto QUIT;

  /* Optimize model */
  error = GRBoptimize(model);
  if (error) goto QUIT;

  /* Write model to 'example_solver.lp' */
  error = GRBwrite(model, "example_solver.lp");
  if (error) goto QUIT;

  /* Capture solution information */
  error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
  if (error) goto QUIT;

  error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &objval);
  if (error) goto QUIT;

  error = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, 4, sol);
  if (error) goto QUIT;

  printf("\nOptimization complete\n");
  if (optimstatus == GRB_OPTIMAL) {
    printf("Optimal objective: %.4e\n", objval);

    printf("a1=%.6f, a2=%.6f, a3=%.6f, a4=%.6f\n", sol[0], sol[1], sol[2], sol[3]);
  } else if (optimstatus == GRB_INF_OR_UNBD) {
    printf("Model is infeasible or unbounded\n");
  } else {
    printf("Optimization was stopped early\n");
  }

QUIT:

  /* Error reporting */
  if (error) {
    printf("ERROR: %s\n", GRBgeterrormsg(env));
    exit(1);
  }

  /* Free model */
  GRBfreemodel(model);

  /* Free environment */
  GRBfreeenv(env);

  return 0;
}