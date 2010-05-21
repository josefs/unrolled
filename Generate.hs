{-# Language QuasiQuotes #-}
{- The idea with this module is that it should generate modules with different
   amount of unrolling of lists. It should take a list of numbers and
   spit out modules which contains these amounts of unrolling

   I'd like to use haskell-src-exts to generate the code.
-}
module Generate where

import Language.Haskell.Exts.Syntax

import Language.Haskell.Exts.QQ

import Language.Haskell.Exts.Pretty

genModule n = Module
              noLoc -- No location, we're generating the code
              (ModuleName ("Data.List.Unrolled" ++ show n)) 
              [] -- No pragmas
              Nothing -- No warning text
              Nothing -- Export list, I export everything
              [] -- Import decls
               -- decls
              [listDeclaration n
              ,elemIndexTy,elemIndex
              ,elemIndicesTy,elemIndices
              ,findTy,find
              ,findIndexTy,findIndex
              ,listToMaybeTy,listToMaybeNil,listToMaybeCons, listToMaybeNCons n
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
listDeclaration n = [$dec| data List a = Nil | Cons a (List a) |]
  where ncons = QualConDecl noLoc [] []
                (ConDecl (Ident "NCons") ((replicate n (UnBangedTy (TyVar (Ident "a"))))
                                          ++ [UnBangedTy (TyApp (TyCon (UnQual (Ident "List"))) (TyVar (Ident "a")))
                                             ]))
-}

-- A couple of definitions that don't depend on the representation of lists
elemIndexTy = [$dec| elemIndex       :: Eq a => a -> List a -> Maybe Int |]
elemIndex   = [$dec| elemIndex x     = findIndex (x==)|]

elemIndicesTy = [$dec| elemIndices     :: Eq a => a -> List a -> List Int |]
elemIndices   = [$dec| elemIndices x   = findIndices (x==)                |]

findTy = [$dec| find            :: (a -> Bool) -> List a -> Maybe a |]
find   = [$dec| find p          = listToMaybe . filter p            |]


findIndexTy = [$dec| findIndex       :: (a -> Bool) -> List a -> Maybe Int |]
findIndex   = [$dec| findIndex p     = listToMaybe . findIndices p         |]

-- Functions depending on the representation of lists
listToMaybeTy   = [$dec| listToMaybe :: List a -> Maybe a    |]
listToMaybeNil  = [$dec| listToMaybe Nil           = Nothing |]
listToMaybeCons = [$dec| listToMaybe (Cons x _)    = Just x  |]
listToMaybeNCons n = 
    FunBind 
    [Match noLoc (Ident "listToMaybe") 
     [PParen (PApp (UnQual (Ident "NCons")) 
                       (PVar (Ident "x") : replicate (n-1) PWildCard))] 
      Nothing 
      (UnGuardedRhs 
       (App (Con (UnQual (Ident "Just"))) (Var (UnQual (Ident "x"))))) (BDecls [])]
