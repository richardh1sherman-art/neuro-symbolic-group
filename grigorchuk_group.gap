# Load the finitely recurrent groups package
LoadPackage("fr");

# Define the binary alphabet {1, 2}
alphabet := 2;

# Define the 4 generators of the Grigorchuk group via wreath recursion
# a swaps the two branches: a = (1, 1)(1,2)
a := State([1, 1], alphabet, (1,2)); 

# b, c, d are defined recursively based on each other:
# b = (a, c), c = (a, d), d = (1, b)
b := State([a, "c"], alphabet);
c := State([a, "d"], alphabet);
d := State([1, "b"], alphabet);

# Tie the recursive string references ("c", "b", "d") to the actual objects
SetRecursion([b, c, d]);

# Create the self-similar group
G := Group(a, b, c, d);

# Example calculation: Check if (a*b)^16 is the identity element
Order(a*b); # Returns 16
