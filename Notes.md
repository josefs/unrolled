% Unrolling Lazy Lists
% Josef Svenningsson
%


# Introduction

Our goal is to make a library of unrolled lists which can be a drop-in 
replacement to the standard Data.List library. In particular we want to
preserve the strictness properties of all functions. At least as far as
possible.

# Unrolling amount

How many times should a list be unrolled?

# Related work

This paper is heavily inspired by Unrolling Lists by Zhong Shao, John
Reppy and Andrew Appel. They present an unrolled representation of lists
in ML. In the discussion below we will refer to that work as UL.

The results presented here differs significantly with UL.

+ First and foremost we are working in a lazy language and UL works in
  a strict language. This difference motivates many of the differences
  presented below.  

+ In UL the lists was represented in such a way that they were
  guaranteed to be unrolled. We have rejected such a representation
  because it would make the lists tail-strict. One of our starting
  goals was to preserve strictness properties compared to standard
  Haskell lists as best we could.

+ They presented an analysis which inferred when a This analysis was
  by necessity incomplete. We have instead chosen to expose unrolled
  lists as a library which give the programmer the choice of when and
  how to use them. This gives more control, transparency and
  predictability but is a little less convenient.


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

# Acknowledgements

Early versions of this paper was written in markdown and processed
with pandoc. I'm greatful to John McFarlane for this wonderful tool.
