{-# Language QuasiQuotes #-}
{- The idea with this module is that it should generate modules with different
   amount of unrolling of lists. It should take a list of numbers and
   spit out modules which contains these amounts of unrolling

-}
module Generate where

import Language.Haskell.Exts.Syntax

import Language.Haskell.Exts.QQ

import Language.Haskell.Exts.Pretty

import Language.Haskell.Exts.Parser

writeModules n = mapM_ writeModule [2..n]

writeModule n = 
    writeFile ("Data/List/Unrolled" ++ show n ++ ".hs")
                  (prettyPrint $ genModule n)

generateModule = putStrLn . prettyPrint . genModule

genModule n = Module
              noLoc -- No location, we're generating the code
              (ModuleName ("Data.List.Unrolled" ++ show n)) 
              [LanguagePragma noLoc [Ident "MagicHash"]] -- Pragmas
              Nothing -- No warning text
              Nothing -- Export list, I export everything
              [ImportDecl noLoc (ModuleName "GHC.Prim") False False
                          Nothing Nothing Nothing
              ,ImportDecl noLoc (ModuleName "GHC.Exts") False False
                          Nothing Nothing Nothing
              ,ImportDecl noLoc (ModuleName "Prelude") True False
                          Nothing (Just (ModuleName "P")) Nothing
              ,ImportDecl noLoc (ModuleName "Prelude") False False 
               Nothing Nothing (Just (True,
               map (IVar . Ident) 
               ["map","take","drop","takeWhile","zip","zipWith","filter"
               ,"sum"] ++
               map (IVar . Symbol)
               ["++"]))
              ,ImportDecl noLoc (ModuleName "Data.List") True False
                          Nothing Nothing
                          (Just (False,[IVar (Ident "sort")]))
              ]
               -- decls
              [listDeclaration n
              ,functorInstance n
              ,monadInstance n
              ,enumClass,enumInstanceInt
              ,eftIntTy,eftIntDec n
              ,toListTy,toListNil,toListCons,toListNCons n
              ,fromListTy,fromListNil,fromListCons
              ,elemIndexTy,elemIndex
              ,elemIndicesTy,elemIndices
              ,findTy,find
              ,findIndexTy,findIndex
              ,findIndicesTy,findIndicesDef n
              ,listToMaybeTy,listToMaybeNil,listToMaybeCons, listToMaybeNCons n
              ,ppTy,ppNil,ppCons,ppNCons n
              ,headTy,headNil,headCons,headNCons n
              ,tailTy,tailNil,tailCons,tailNCons n
              ,nullTy,nullNil,nullCons,nullNCons n
              ,lengthTy,lengthDec,lenTy,lenNil,lenCons,lenNCons n
              ,mapTy,mapNil,mapCons,mapNCons n
              ,sumSpec,sumLoopSpec
              ,sumTy,sumDec,sumLoopTy,sumLoopNil,sumLoopCons,sumLoopNCons n
              ,sumBadTy,sumBadNil,sumBadCons,sumBadNCons n
              ,takeTy,takeZero,takeNil,takeCons,takeNCons n
              ,dropTy,dropZero,dropNil,dropCons,dropNCons n
              ,splitAtDef
              ,takeWhileTy,takeWhileNil,takeWhileCons,takeWhileNCons n
              ,filterTy,filterNil,filterCons,filterNCons n

              ,zipNil1,zipNil2,zipCons,zipConsN1 n,zipConsN2 n,zipNCons n
              ,zipWithTy, zipWithNil1,zipWithNil2,zipWithCons
              ,zipWithConsN1 n,zipWithConsN2 n,zipWithNCons n

              ,sortTy,sortDec

              ,errEmptyTy,errEmptyDec n
              ]

noLoc = SrcLoc "Unknown" 0 0

listDeclaration n = -- data List a = Nil | Cons a (List a) | NCons a a (List a)
  DataDecl noLoc DataType [] 
               (Ident "List")
               [UnkindedVar (Ident "a")]
               [QualConDecl noLoc [] [] (ConDecl (Ident "Nil") [])
               ,QualConDecl noLoc [] [] (ConDecl (Ident "Cons") 
                                         [UnBangedTy (TyVar (Ident "a"))
                                         ,UnBangedTy (TyApp (TyCon (UnQual (Ident "List"))) (TyVar (Ident "a")))
                                         ])
               ,QualConDecl noLoc [] [] (ConDecl (Ident "NCons")
                                         (map UnBangedTy $
                                          replicate n (TyVar (Ident "a")) ++
                                          [TyApp (TyCon (UnQual (Ident "List"))) (TyVar (Ident "a"))]))
               ]
               [(UnQual (Ident "Show"),[])
               ,(UnQual (Ident "Eq"),[])
               ,(UnQual (Ident "Ord"),[])
               ]

{- This gives a very weird error message.
listDeclaration n = [dec| data List a = Nil | Cons a (List a) |]
  where ncons = QualConDecl noLoc [] []
                (ConDecl (Ident "NCons") ((replicate n (UnBangedTy (TyVar (Ident "a"))))
                                          ++ [UnBangedTy (TyApp (TyCon (UnQual (Ident "List"))) (TyVar (Ident "a")))
                                             ]))
-}

functorInstance n =
    InstDecl noLoc  [] (UnQual (Ident "Functor")) 
                 [TyCon (UnQual (Ident "List"))] $
             map InsDecl $
             [ [dec| fmap f Nil = Nil |]
             , [dec| fmap f (Cons x xs) = Cons (f x) (fmap f xs) |]
             , functionCl "fmap" [varp "f",nconsp "x" n "xs"] $
                 apps (constr "NCons" : [apps [var "f",var ("x" ++ show i)] 
                                      | i <- [1..n]] ++ 
                                      [apps [var "fmap",var "f",var "xs"]])
             ]

monadInstance n =
    InstDecl noLoc  [] (UnQual (Ident "Monad")) 
                 [TyCon (UnQual (Ident "List"))] $
             map InsDecl $
             [ [dec| return a = Cons a Nil |]
             , [dec| Nil >>= f = Nil |]
             , [dec| (Cons x xs) >>= f = f x ++ (xs >>= f) |]
             , functionCl "(>>=)" [nconsp "x" n "xs",varp "f"] $
               pps [apps [var "f",var ("x" ++ show i)] | i <- [1..n]]
                   (Paren (InfixApp (var "xs") 
                                    (QVarOp (UnQual (Symbol ">>=")))
                                    (var "f")))
             ]

enumClass = [dec|
class Enum a => EnumList a where
    enumFromL              :: a -> List a
    enumFromL              =  fromList . enumFrom
    enumFromThenL          :: a -> a -> List a
    enumFromThenL      f t =  fromList $ enumFromThen f t
    enumFromToL            :: a -> a -> List a
    enumFromToL        f t =  fromList $ enumFromTo f t
    enumFromThenToL        :: a -> a -> a -> List a
    enumFromThenToL f th t = fromList $ enumFromThenTo f th t
|]

enumInstanceInt = InstDecl noLoc [] (UnQual (Ident "EnumList")) [TyCon (UnQual (Ident "Int"))]
                  [InsDecl $
                           functionCl "enumFromToL"
                                          [PApp (UnQual (Ident "I#")) [varp "i"]
                                          ,PApp (UnQual (Ident "I#")) [varp "j"]]
                           (apps [var "eftInt",var "i",var "j"])
                  ]

eftIntTy    = [dec| eftInt :: Int# -> Int# -> List Int |]
eftIntDec n = [dec| eftInt x0 y | x0 ># y   = Nil
                                | otherwise = $(eftIntLoop n)
                 |]

eftIntLoop n = Let (BDecls [functionCl "go" [varp "x"] $
                  Case (InfixApp x (QVarOp (UnQual (Symbol "-#"))) y) $
                           [ Alt noLoc (PLit (PrimInt i))
                                     (UnGuardedAlt (conses [ apps [constr "I#"
                                                                  ,x +#+  (Lit (PrimInt j))]
                                                           | j <- [0..i] ] (constr "Nil")))
                                     (BDecls [])
                           | i <- [0..toInteger n - 2] ]
                           ++
                           [Alt noLoc PWildCard
                                    (UnGuardedAlt (apps ([constr "NCons"] ++
                                                         [apps [constr "I#"
                                                               ,x +#+  (Lit (PrimInt i)) ]
                                                         | i <- [0..toInteger n - 1] ] ++
                                                         [apps [var "eftInt"
                                                               ,x +#+ (Lit (PrimInt (toInteger n)))
                                                               ,y
                                                               ]])))
                            (BDecls [])]])
          (apps [var "go", var "x0"])
  where x +#+ y = InfixApp x (QVarOp (UnQual (Symbol "+#"))) y
        x = var "x"
        y = var "y"

toListTy   = [dec| toList :: List a -> [a]            |]
toListNil  = [dec| toList Nil         = []            |]
toListCons = [dec| toList (Cons x xs) = x : toList xs |]
toListNCons n =
    functionCl "toList" [nconsp "x" n "xs"] $
               colons [var ("x" ++ show i) | i <- [1..n]]
                          (apps [var  "toList",var "xs"])

fromListTy   = [dec| fromList :: [a] -> List a              |]
fromListNil  = [dec| fromList [] = Nil                      |]
fromListCons = [dec| fromList (x:xs) = Cons x (fromList xs) |]
-- Should we have an NCons case?
{-
fromListNCons n =
    functionCl "fromList" [PInfixApp (varp "x") (] $
-}

-- cnses ls exp = foldr (\x a -> InfixApp x (UnQual (Symbol ":")) a) exp ls


-- A couple of definitions that don't depend on the representation of lists
elemIndexTy = [dec| elemIndex       :: Eq a => a -> List a -> Maybe Int |]
elemIndex   = [dec| elemIndex x     = findIndex (x==)|]

elemIndicesTy = [dec| elemIndices     :: Eq a => a -> List a -> List Int |]
elemIndices   = [dec| elemIndices x   = findIndices (x==)                |]

findTy = [dec| find            :: (a -> Bool) -> List a -> Maybe a |]
find   = [dec| find p          = listToMaybe . filter p            |]


findIndexTy = [dec| findIndex       :: (a -> Bool) -> List a -> Maybe Int |]
findIndex   = [dec| findIndex p     = listToMaybe . findIndices p         |]

-- Functions depending on the representation of lists
listToMaybeTy   = [dec| listToMaybe :: List a -> Maybe a    |]
listToMaybeNil  = [dec| listToMaybe Nil           = Nothing |]
listToMaybeCons = [dec| listToMaybe (Cons x _)    = Just x  |]
listToMaybeNCons n = 
    FunBind 
    [Match noLoc (Ident "listToMaybe") 
     [PParen (PApp (UnQual (Ident "NCons")) 
                       ((varp "x") : replicate n PWildCard))]
      Nothing 
      (UnGuardedRhs
       (App (constr "Just") (var "x"))) (BDecls [])]

u = error "Unimplemented"


ppTy   = [dec| (++) :: List a -> List a -> List a  |]
ppNil  = [dec| Nil       ++ l = l                  |]
ppCons = [dec| Cons x l1 ++ l2 = Cons x (l1 ++ l2) |]
ppNCons n = 
    FunBind
    [Match noLoc (Ident "(++)")
     [PParen (PApp (UnQual (Ident "NCons"))
                   ([PVar (Ident ("x" ++ show i)) | i <- [1..n]]
                   ++ [PVar (Ident "l1")])
             )
     ,varp "l2"]
     Nothing
     (UnGuardedRhs
      (apps (constr "NCons" :
                     [var ("x" ++ show i) | i <- [1..n]]
                     ++ [apps [var "(++)"
                              ,var "l1"
                              ,var "l2"]])))
     (BDecls [])]

headTy   = [dec| head :: List a -> a |]
headNil  = [dec| head Nil = errorEmptyList "head" |]
headCons = [dec| head (Cons x _) = x |]
headNCons n = 
    FunBind
    [Match noLoc (Ident "head")
     [PParen (PApp (UnQual (Ident "NCons"))
                   ([varp ("x" ++ show i) | i <- [1..n]]
                   ++ [varp "l1"])
             )]
     Nothing
     (UnGuardedRhs (var "x1"))
     (BDecls [])]

last = u

tailTy   = [dec| tail :: List a -> List a |]
tailNil  = [dec| tail Nil = errorEmptyList "tail" |]
tailCons = [dec| tail (Cons _ l) = l |]
tailNCons n =
    functionCl "tail" [nconsp "x" n "xs"] (conses
                                   [ var ("x" ++ show i)
                                         | i <- [2..n] ] 
                                   (var "xs"))

init = u

nullTy   = [dec| null :: List a -> Bool |]
nullNil  = [dec| null Nil = True        |]
nullCons = [dec| null (Cons _ _) = False |]
nullNCons n =
    functionCl "null" [nconsp "x" n "xs"] (constr "False")

lengthTy  = [dec| length :: List a -> Int  |]
lengthDec = [dec| length l = I# (len l 0#) |]

lenTy = [dec| len :: List a -> Int# -> Int# |]
lenNil = [dec| len Nil a# = a# |]
lenCons = [dec| len (Cons _ xs) a# = len xs (a# +# 1#) |]
lenNCons n =
    functionCl "len" [nconsp "x" n "xs",varp "a#"] $
               apps [var "len",var "xs"
                    ,InfixApp (var "a#") (QVarOp (UnQual (Symbol "+#")))
                                  (Lit (PrimInt (toInteger n)))]

mapTy   = [dec| map :: (a -> b) -> List a -> List b     |]
mapNil  = [dec| map f Nil = Nil                         |]
mapCons = [dec| map f (Cons a l) = Cons (f a) (map f l) |]
mapNCons n =
    functionCl "map" [PVar (Ident "f"),nconsp "x" n "xs"]
               (apps ([constr "NCons"]
                      ++ [ App (var "f")
                               (var ("x" ++ show i))
                         | i <- [1..n] ]
                      ++ [apps [var "map",var "f",var "xs"]]))

reverse = u

-- Maybe skip these four initially?
intersperse = u
transpose = u
subsequence = u
permutations = u

foldl = u

foldl' = u

foldl1d = u

foldl1' = u

foldrd = u

foldr1d = u

concat = u

concatMap = u

and = u

or = u

any = u

all = u

sumBadTy   = [dec| sumBad :: (Num a) => List a -> a |]
sumBadNil  = [dec| sumBad Nil = 0 |]
sumBadCons = [dec| sumBad (Cons x xs) = x + (sumBad xs) |]
sumBadNCons n =
    functionCl "sumBad" [nconsp "x" n "xs"] $
               pluses [var ("x" ++ show i) | i <- [1..n]]
                      (apps [var "sumBad", var "xs"])


sumSpec = [dec| {-# SPECIALISE sum :: List Int -> Int #-} |]

sumTy  = [dec| sum :: (Num a) => List a -> a |]
sumDec = [dec| sum l = sumLoop l 0           |]

sumLoopSpec = [dec| {-# SPECIALISE sumLoop :: List Int -> Int -> Int #-} |]

sumLoopTy   = [dec| sumLoop :: Num a => List a -> a -> a |]
sumLoopNil  = [dec| sumLoop Nil a = a                    |]
sumLoopCons = [dec| sumLoop (Cons x xs) a = let s = a+x in s `seq` sumLoop xs s |]
sumLoopNCons n =
    functionCl "sumLoop" [nconsp "x" n "xs", varp "a"] $
               Let (BDecls [functionCl "s" [] 
                            (pluses [var ("x" ++ show i) | i <- [1..n]] (var "a"))
                           ])
                   (apps [var "seq", var "s", apps [var "sumLoop"
                                                   ,var "xs"
                                                   ,var "s"]
                         ])
sum = u

product = u

maximum = u

minimum = u

-- Skip these for now
scanl = u
scanl1 = u
scanr = u
scanr1 = u

-- Skip for now
mapAccumL = u
mapAccumR = u

iterate = u

repeat = u

replicated = u

cycle = u

unfoldr = u

-- This function could use some optimization.
-- For instance, we only want to check that n <= 0 once.
takeTy   = [dec| take :: Int -> List a -> List a           |]
takeZero = [dec| take n _ | n <= 0 = Nil                   |]
takeNil  = [dec| take _ Nil        = Nil                   |]
takeCons = [dec| take n (Cons a l) = Cons a (take (n-1) l) |]
takeNCons n =
    functionCl "take" [varp "n",nconsp "x" n "xs"] $
             If (InfixApp (var "n") (op ">=") (int n))
                (apps (constr "NCons" : [ var ("x" ++ show i) | i <- [1..n] ]
                       ++ [apps [var "take",InfixApp (var "n") (op "-") (int n),var "xs"]]))
                (ifChain (n-1) (\i -> InfixApp (var "n") (op "==") (int i))
                               (\i -> conses [var ("x" ++ show j)| j <- [1..i]] 
                                    (constr "Nil"))
                               (constr "Nil"))

-- Could use some optimization. See Data.List
dropTy   = [dec| drop :: Int -> List a -> List a  |]
dropZero = [dec| drop n l | n <= 0 = l            |]
dropNil  = [dec| drop n Nil        = Nil          |]
dropCons = [dec| drop n (Cons x l) = drop (n-1) l |]
dropNCons n =
    functionCl "drop" [varp "n",nconsp "x" n "xs"] $
               If (InfixApp (var "n") (op ">=") (int n))
                  (apps [var "drop", InfixApp (var "n") (op "-") (int n), var "xs"])
                  (ifChain (n-1) (\i -> InfixApp (var "n") (op "==") (int i))
                                 (\i -> conses [var ("x" ++ show j) 
                                               | j <- [i+1..n]] (constr "Nil"))
                                 (constr "Nil"))

--splitAtTy = [dec| splitAt :: Int -> List a -> (List a, List a) |]
splitAtDef = [dec| splitAt n ls = (take n ls, drop n ls) |]

takeWhileTy   = [dec| takeWhile :: (a -> Bool) -> List a -> List a |]
takeWhileNil  = [dec| takeWhile _ Nil         = Nil                |]
takeWhileCons = [dec| takeWhile p (Cons x xs) | p x = Cons x (takeWhile p xs)
                                              | otherwise = Nil |]
-- Should be made lazier!
takeWhileNCons n =
    functionCl "takeWhile" [varp "p", nconsp "x" n "xs"] $
               ifChainRev 1 n 
                 (\i -> apps [var "not ", apps [var "p"
                                               ,var ("x" ++ show i)]])
                 (\i -> conses [var ("x" ++ show j)
                               | j <- [1..i-1]] (constr "Nil"))
                 (conses [var ("x" ++ show i)
                         | i <- [1..n]] (apps [var "takeWhile"
                                              ,var "p"
                                              ,var "xs"]))

dropWhile = u

-- Same problem here with the type signature
--spanTy = [dec| span :: (a -> Bool) -> List a -> (List a, List a) |]
spanNil  = [dec| span _ Nil = (Nil,Nil) |]
spanCons = [dec| span p xs@(Cons x xs') 
               | p x       = let (ys,zs) = span p xs' in (x:ys,zs)
               | otherwise = (Nil,xs) |]
{-
spanNCons n =
    function "span" [varp "p", nconsp "x" n 1]
             
  span p xs@(NCons x1 x2 x3 xs') =
      if not (p x1) then (Nil,xs)
      else if not (p x2) then (Cons x1 Nil,Cons x2 (Cons x3 xs'))
           else if not (p x3) then (Cons x1 (Cons x2 Nil),Cons x3 xs')
                else let (ys,zs) = span p xs'
                     in (Cons x1 (Cons x2 (Cons x3 ys)),zs)
-}
span = u

break = u

-- Skip these for now
stripPrefix = u
group = u

inits = u

tails = u

isPrefixOf = u

isSuffixOf = u

isInfixOf = u

elem = u

notElem = u

lookup = u

filterTy   = [dec| filter :: (a -> Bool) -> List a -> List a        |]
filterNil  = [dec| filter _ Nil = Nil                               |]
filterCons = [dec| filter p (Cons x xs) | p x = Cons x (filter p xs)
                                        | otherwise = filter p xs   |]
filterNCons n =
    functionCl "filter" [varp "p",nconsp "x" n "xs"] $
               apps [var "filter",var "p"
                    ,conses [var ("x" ++ show i) | i <- [1..n]] (var "xs")]

partition = u

(!!) = u

findIndicesTy = [dec| findIndices :: (a -> Bool) -> List a -> List Int |]
findIndicesDef n =
    functionCl "findIndices" [varp "p", varp "xs"] $
               Let (BDecls 
                    [functionCl "loop" [PWildCard,constrp "Nil" []] 
                                    (constr "Nil")
                    ,functionCl "loop" [varp "n",consp "x" "xs"] $
                                If (apps [var "p",var "x"])
                                   (apps [constr "Cons"
                                         ,apps [constr "I#",var "n"]
                                         ,apps [var "loop"
                                               ,InfixApp (var "n")
                                                         (QVarOp (UnQual (Symbol "+#")))
                                                         (Lit (PrimInt 0))
                                               ,var "xs"]
                                         ])
                                   (apps [var "loop"
                                         ,InfixApp (var "n")
                                                   (QVarOp (UnQual (Symbol "+#")))
                                                   (Lit (PrimInt 0))
                                         ,var "xs"])
                    ,functionCl "loop" [varp "n",nconsp "x" n "xs"] $
                                apps [var "loop"
                                     ,var "n"
                                     ,conses [var ("x" ++ show i) 
                                             | i <- [1..n]]
                                             (var "xs")
                                     ]
                    ])
                   (apps [var "loop", Lit (PrimInt 0), var "xs"])

-- I get really weird error messages involving interface files when I uncomment
-- the zipTy line.
--zipTy = [dec| zip :: List a -> List b -> List (a,b) |]
zipNil1 = [dec| zip Nil _ = Nil |]
zipNil2 = [dec| zip _ Nil = Nil |]
zipCons = [dec| zip (Cons x xs) (Cons y ys) = Cons (x,y) (zip xs ys) |]

zipConsN1 n =
    functionCl "zip" [consp "x" "xs",nconsp "y" n "ys"]
               (apps [constr "Cons", Tuple [var "x",var "y1"],
                           apps [var "zip", var "xs", conses [var ("y" ++ show i) | i <- [2..n] ] (var "ys") ]])
zipConsN2 n =
    functionCl "zip" [nconsp "x" n "xs", consp "y" "ys"]
               (apps [constr "Cons", Tuple [var "x1",var "y"],
                           apps [var "zip", conses [var ("x" ++ show i) | i <- [2..n] ] (var "xs") , var "ys" ]])
zipNCons n =
    functionCl "zip" [nconsp "x" n "xs", nconsp "y" n "ys"]
               (apps (constr "NCons" : [ Tuple [var ("x" ++ show i)
                                             ,var ("y" ++ show i)]
                                      | i <- [1..n] ]
                      ++ [apps [var "zip", var "xs", var "ys"]]))

zip3 = u

-- Skip zip4 - zip 7 for now

zipWithTy   = [dec| zipWith :: (a -> b -> c) -> List a -> List b -> List c |]
zipWithNil1 = [dec| zipWith f Nil _ = Nil |]
zipWithNil2 = [dec| zipWith f _ Nil = Nil |]
zipWithCons = [dec| zipWith f (Cons x xs) (Cons y ys) = Cons (f x y) (zipWith f xs ys) |]

zipWithConsN1 n =
    functionCl "zipWith" [varp "f", consp "x" "xs",nconsp "y" n "ys"]
               (apps [constr "Cons", apps [var "f", var "x",var "y1"],
                           apps [var "zipWith", var "f", var "xs", conses [var ("y" ++ show i) | i <- [2..n] ] (var "ys") ]])
zipWithConsN2 n =
    functionCl "zipWith" [varp "f", nconsp "x" n "xs", consp "y" "ys"]
               (apps [constr "Cons", apps [var "f", var "x1",var "y"],
                           apps [var "zipWith", var "f", conses [var ("x" ++ show i) | i <- [2..n] ] (var "xs") , var "ys" ]])
zipWithNCons n =
    functionCl "zipWith" [varp "f", nconsp "x" n "xs", nconsp "y" n "ys"]
               (apps (constr "NCons" : [ apps [var "f"
                                            ,var ("x" ++ show i)
                                            ,var ("y" ++ show i)]
                                      | i <- [1..n] ]
                      ++ [apps [var "zipWith", var "f", var "xs", var "ys"]]))

zipWith3 = u

-- Skip zipWith4-7

unzip = u

-- Skip unzip3-7

lines = u

words = u

unwords = u

nub = u

delete = u

(\\) = u

union = u

intersect = u

sortTy  = [dec| sort :: Ord a => List a -> List a       |]
sortDec = [dec| sort = fromList . Data.List.sort . toList |]

insert = u

-- Skip these for no
nubBy = u
deleteBy = u
deleteFirstBy = u
unionBy = u
intersectBy = u
groupBy = u

sortBy = u

insertBy = u

maximumBy = u

minimumBy = u

-- Skip generic* functions for now

errEmptyTy    = [dec| errorEmptyList :: String -> a |]
errEmptyDec n = [dec| errorEmptyList fun = error ( $(Lit $ String $ "Data.List.Unrolled" ++ show n ++ ".") P.++ fun P.++ ": empty list") |]


-- Helper functions for building syntax

apps = foldl1 App

functionCl name pat exp =
    FunBind
    [Match noLoc (Ident name)
     pat
     Nothing
     (UnGuardedRhs exp)
     (BDecls [])]

varp name = PVar (Ident name)

nconsp e n l = PParen (PApp (UnQual (Ident "NCons"))
                       ([PVar (Ident (e ++ show i)) | i <- [1..n]]
                        ++ [PVar (Ident l)]))

consp e l = PParen (PApp (UnQual (Ident "Cons")) [varp e,varp l])

constrp name ps = PApp (UnQual (Ident name)) ps

conses ls exp = foldr (\x a -> apps [Con (UnQual (Ident "Cons"))
                                    ,x,a]) exp ls

pps ls exp = foldr (\x a -> InfixApp x (QVarOp (UnQual (Symbol "++"))) a) exp ls

colons ls exp = foldr (\x a -> InfixApp x (QConOp (UnQual (Symbol ":"))) a) exp ls

pluses ls exp = foldr (\x a -> InfixApp x (QVarOp (UnQual (Symbol "+"))) a) exp ls

var name = Var (UnQual (Ident name))

constr name = Con (UnQual (Ident name))

op name = QVarOp (UnQual (Symbol name))

int i = Lit (Int (toInteger i))

ifChain :: Num a => a -> (a -> Exp) -> (a -> Exp) -> Exp -> Exp
ifChain 0 _ _ e = e
ifChain n p f e = If (p n)
                     (f n)
                     (ifChain (n-1) p f e)

ifChainRev i n p f e | i > n = e
ifChainRev i n p f e = If (p i)
                          (f i)
                          (ifChainRev (i+1) n p f e)
