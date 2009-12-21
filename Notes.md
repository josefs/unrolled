% Unrolled Lazy Lists
% Josef Svenningsson
%


# Introduction

Our goal is to make a library of unrolled lists which can be a drop-in 
replacement to the standard Data.List library. In particular we want to
preserve the strictness properties of all functions. At least as far as
possible.

# Fusion

Fusing unrolled lists.

# Adaptive Unrolled Lists.

Don Stewart has a library for lists which specializes their
representation depending on what type their elements has. 

Parametric polymorphism requires that all types have the same
representation if we are to have a single implementation of the
polymorphic function or data type. This imposes a performance penalty,
especially for primitive types such as integers which can be
represented much more succintly unboxed.
