# Encoding for Continuous Thought Using Self-Similar Groups

## Introduction
The purpose of this document is to describe the API with the neuro-symbolic engine and to explain problem encoding for that engine. This document serves as a standard architectural blueprint and reference guide for system designers and future LLMs across various applications.

---

## 1. Architectural Foundations & Invariants

### 1.1 Model Theory of First-Order Logic (Wilson 2015)
The engine discards arbitrary flat relational tables in favor of self-similar branch groups acting faithfully on infinite rooted trees (T). Vertices at every level possess identical finite valency. The distance of a vertex v from the root vertex v₀ is defined as its level, and the set \(L_n\) of vertices of level n forms the n-th layer of T. 

The tree structure can be completely reconstructed from its partial order, where u ≤ v if and only if the simple path from u to v₀ passes through v. Each vertex v represents the root of an independent subtree \(T_v = \{u \mid u \le v\}\).

### 1.2 Self-Similar Subgroup Domain Isolation
To model distinct namespaces, independent family bloodlines, and categorical filters, the engine isolates operations into self-similar subgroups. A subgroup \(H \le G\) is self-similar if for every element \(h \in H\), its restriction on any child subtree lands entirely back within H. 

The architecture relies on two core model-theoretic subgroups:
1. **The Rigid Stabilizer Subgroup (\(rst_G(v)\)):** Contains elements that fix every vertex outside the target subtree \(T_v\), acting as an isolated local predicate database.
2. **The Layer Stabilizer Subgroup (\(rst_G(n)\)):** The direct product of all vertex stabilizers at a given layer \(L_n\):
   \[rst_G(n) = \prod_{v \in L_n} rst_G(v)\]

Because these subgroups act on disjoint branches, their relational interaction forms a direct product. The engine evaluates their intersection via a distributive lattice meet (\(\wedge\)), ensuring operations inside one domain cannot leak into or corrupt the state coordinates of an adjacent domain.

---

## 2. API & Problem Encoding Blueprint

### 2.1 The Universal Implementation Language
Every application in the framework compiles down to a single universal language: **Relational Algebra mapped to Tensor Contractions**. Continuous, fuzzy neural network optimization processes sit strictly outside this language as an external phase handled natively by Python. The core algebra remains mathematically closed, clean, and rigidly bounded.

| Abstraction Layer | Computational Paradigm | Primitive Operator | Implementation Vector |
| :--- | :--- | :--- | :--- |
| **Fuzzy / Continuous** | Superposition Space | Cross-Attention Logits | Python / Neural Network |
| **Universal Algebraic** | Relational Algebra | Kronecker Matrix Contracts | `torch.matmul()`, `torch.kron()` |
| **Formal Symbolic** | First-Order Logic | Definite Horn Clauses | Prolog `.pl` Files / Z3 SMT |

### 2.2 Core Module API Definitions

#### 1. Core Algebraic Compiler (`TrueBottomUpGrigorchukCompiler`)
Translates abstract mathematical code structures and recursive group strings bottom-up into hardware-accelerated Kronecker matrix blocks.
*   `__init__(max_depth=5)`: Pre-allocates tensor blocks down to the target tree depth.
*   `_build_tree_layer(current_depth)`: Executes the recursive bottom-up invocation loop. Returns a four-tuple tensor layer `(a, b, c, d)`.
*   `compute_word_order(word_matrix, max_order=32)`: Evaluates identity collapse metrics. Returns an integer tracking the periodic sequence limit, or `-1` on a timeout.

#### 2. Inverse Logic Decoder (`TensorToHornDecoder`)
Deconstructs continuous multi-scale GPU matrices back into readable first-order Prolog logic primitives.
*   `decode_block_to_predicate(matrix_slice, depth_level)`: Evaluates a 2x2 hardware slice. Returns `"branch_swap"`, `"identity_invariant"`, or `"recursive_restriction"`.
*   `synthesize_horn_clause(generator_name, generator_matrix)`: Compiles an active operator matrix into a definitive definite Horn clause string.
*   `export_knowledge_base(clauses_list, output_filename="grigorchuk_rules.pl")`: Flushes compiled relational array memory out into a readable Prolog `.pl` file.

#### 3. Automated SMT Theorem Prover (`NeuroSymbolicTheoremProver`)
Parses relational knowledge blocks and pushes them into an automated SMT solver context to verify system invariants.
*   `ingest_knowledge_base()`: Uses robust regular expressions (`re.search`) to match Prolog predicates and assert equivalent structural constraints onto Z3 boolean variable fields.
*   `execute_proof()`: Injects the negation of the hypothesis statement into the solver context. Runs a `.check()` verification pass to enforce proofs via absolute contradiction (`unsat`).

#### 4. Path Trajectory Tracker (`SelfSimilarProofTracer`)
Traces active matrix path indices step-by-step across GPU memory matrices to extract explicit mathematical proof logs.
*   `int_to_binary_path(index_value)`: Translates a raw matrix coordinate row/column integer back into a readable binary tree path string (`"left -> right"`).
*   `trace_word_annihilation(word_name, word_matrix, max_steps=32)`: Multiplies sequence matrices, computes L₂ distance metrics, and outputs exact `Row -> Col` trajectory steps to eliminate black-box limitations.

#### 5. Bidirectional Reversible Substrate (`ReversibleTreeEngine`)
Enforces information-theoretic and structural reversibility to eliminate database caching bottlenecks.
*   `execute_forward_transformation(initial_state, operator_matrix)`: Fires an accelerated parallel tensor multiplication pass forward down the branches.
*   `execute_reverse_backtrack(current_state, operator_matrix)`: Computes the exact group matrix inverse (g⁻¹) to step right backward, fully reconstructing the original state vector with a perfect `0.000000` L₂ distance delta.

---

## 3. Application Mappings & Generator Encodings

### 3.1 First-Order Logic Poset Trees (`branch_group_fol.py`, `branch_group_induction.py`)
*   **Generators:** Structural tree-stabilizer generators defining path descents over a binary split hierarchy.
*   **Vector Organization:** Organized as dense, 2-adic discrete coordinate sequence arrays.
*   **Algebraic Call:** `torch.matmul()` executes fact chaining (Hypothetical Syllogism) via matrix-by-matrix products, and fact deduction (Modus Ponens) via matrix-by-vector products.
*   **Encoding Example:**
```python
# Vector Structure: [ Namespace_ID, Generational_Depth, Attribute_Bit ]
self.paths["Alice"]   = [0, 2, 0]
self.paths["Bob"]     = [0, 3, 1]
self.paths["Charlie"] = [0, 4, 0]

# Poset Intersection Meet (Definition 2.2) via Zip Reduction
def compute_lattice_meet(self, path_a, path_b):
    meet = []
    for a, b in zip(path_a, path_b):
        if a == b: meet.append(a)
        else: break
    return meet
```

### 3.2 Automated Predicate Learning via Inductive Closures (`learn_new_predicates.py`)
*   **Generators:** Lattice Meet (\(\wedge\)) and Join (\(ee\)) Operators governed by strict partial orders.
*   **Vector Organization:** Coordinate paths mapping target domain interactions.
*   **Algebraic Call:** Step-by-step element-wise array matching to compute structural logical closures.
*   **Encoding Example:**
```python
# Inducts the completely unknown 'father' predicate by evaluating the structural closure 
# of pre-existing base matrix conditions: father(X, Y) :- parent(X, Y), is_male(X).
has_parent_link = self.evaluate_known_parent(entity_x, entity_y)
has_male_link = self.evaluate_known_male(entity_x)
predicate_strength = 1.0 if (has_parent_link and has_male_link) else 0.0
```

### 3.3 Meta-Interpretive Learning & Metarule Discovery (`discover_metarules.py`)
*   **Generators:** Wreath-Product Recursive Chaining Metarule Templates.
*   **Vector Organization:** Multi-layer rule configuration maps tracking top-level and sub-block behavior.
*   **Algebraic Call:** Semicolon branch-action matching across distinct tree slices.
*   **Encoding Example:**
```python
# Synthesizes an abstract higher-order grammar rule by analyzing structural symmetries 
# across Level 1 (top-branch swaps) and Level 2 (subtree embeddings):
metarule = "meta_template_1(R, P, Q) :- active_swap(P, level_1), embedded_restriction(Q, P, level_2)."
```

### 3.4 Continuous Thought Superposition Core (`run_live_inference.py`)
*   **Superposition Processor:** The **Neural Rule Inducer** acts as the soft, fuzzy optimization computer.
*   **Vector Organization:** Dense literal and clause statistics vectors tracking class-conditional positive/negative frequency rates (P⁺/P⁻), structural entropy (H), and co-occurrence matrices (C).
*   **Algebraic Call:** Continuous cross-attention slot decoding evaluated via soft T-norm logic gates (z and w) in Python.
*   **Encoding Example:**
```python
# Python extracts the soft continuous logits from the Neural Rule Inducer superposition space 
# and hands them over to the discrete group language for structural discretization:
logits = model(input_tensor)
probabilities = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
p_g1 = probabilities[dataset.op_vocab.get("OP_G1", 1)]
```
