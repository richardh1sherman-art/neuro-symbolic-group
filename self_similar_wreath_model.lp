\* Universal Neuro-Symbolic Self-Similar Group Optimization Model *\

Minimize
   1 c_step_0_G1 + 2 c_step_0_G2 + 1 c_step_1_G1 + 2 c_step_1_G2 + 1 c_step_2_G1 + 2 c_step_2_G2 + 1 c_step_3_G1 + 2 c_step_3_G2

Subject To
  Exclusivity_Step_0: c_step_0_G1 + c_step_0_G2 = 1
  Exclusivity_Step_1: c_step_1_G1 + c_step_1_G2 = 1
  Exclusivity_Step_2: c_step_2_G1 + c_step_2_G2 = 1
  Exclusivity_Step_3: c_step_3_G1 + c_step_3_G2 = 1

  \* Boundary Trajectory Vectors *\
  Initial_State_Bound: state_0 = 6
  Target_State_Bound: state_4 = 0

  \* Algebraic State Alignment Rules *\
  Algebraic_Transition_Step_0: state_1 - state_0 = -4
  Algebraic_Transition_Step_1: state_2 - state_1 = 1
  Algebraic_Transition_Step_2: state_3 - state_2 = -2
  Algebraic_Transition_Step_3: state_4 - state_3 = -1

Bounds
  0 <= state_0 <= 15
  0 <= state_1 <= 15
  0 <= state_2 <= 15
  0 <= state_3 <= 15
  0 <= state_4 <= 15

Binary
  c_step_0_G1
  c_step_0_G2
  c_step_1_G1
  c_step_1_G2
  c_step_2_G1
  c_step_2_G2
  c_step_3_G1
  c_step_3_G2

End
