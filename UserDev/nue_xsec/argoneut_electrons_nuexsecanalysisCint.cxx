// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME argoneut_electrons_nuexsecanalysisCint

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "RConfig.h"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// Since CINT ignores the std namespace, we need to do so in this file.
namespace std {} using namespace std;

// Header files passed as explicit arguments
#include "Stage1Efficiency.h"

// Header files passed via #pragma extra_include

namespace argoana {
   namespace ROOT {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *argoana_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("argoana", 0 /*version*/, "Stage1Efficiency.h", 36,
                     ::ROOT::Internal::DefineBehavior((void*)0,(void*)0),
                     &argoana_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *argoana_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}

namespace ROOT {
   static TClass *argoanacLcLStage1Efficiency_Dictionary();
   static void argoanacLcLStage1Efficiency_TClassManip(TClass*);
   static void *new_argoanacLcLStage1Efficiency(void *p = 0);
   static void *newArray_argoanacLcLStage1Efficiency(Long_t size, void *p);
   static void delete_argoanacLcLStage1Efficiency(void *p);
   static void deleteArray_argoanacLcLStage1Efficiency(void *p);
   static void destruct_argoanacLcLStage1Efficiency(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::argoana::Stage1Efficiency*)
   {
      ::argoana::Stage1Efficiency *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::argoana::Stage1Efficiency));
      static ::ROOT::TGenericClassInfo 
         instance("argoana::Stage1Efficiency", "Stage1Efficiency.h", 42,
                  typeid(::argoana::Stage1Efficiency), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &argoanacLcLStage1Efficiency_Dictionary, isa_proxy, 4,
                  sizeof(::argoana::Stage1Efficiency) );
      instance.SetNew(&new_argoanacLcLStage1Efficiency);
      instance.SetNewArray(&newArray_argoanacLcLStage1Efficiency);
      instance.SetDelete(&delete_argoanacLcLStage1Efficiency);
      instance.SetDeleteArray(&deleteArray_argoanacLcLStage1Efficiency);
      instance.SetDestructor(&destruct_argoanacLcLStage1Efficiency);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::argoana::Stage1Efficiency*)
   {
      return GenerateInitInstanceLocal((::argoana::Stage1Efficiency*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const ::argoana::Stage1Efficiency*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *argoanacLcLStage1Efficiency_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::argoana::Stage1Efficiency*)0x0)->GetClass();
      argoanacLcLStage1Efficiency_TClassManip(theClass);
   return theClass;
   }

   static void argoanacLcLStage1Efficiency_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_argoanacLcLStage1Efficiency(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::argoana::Stage1Efficiency : new ::argoana::Stage1Efficiency;
   }
   static void *newArray_argoanacLcLStage1Efficiency(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::argoana::Stage1Efficiency[nElements] : new ::argoana::Stage1Efficiency[nElements];
   }
   // Wrapper around operator delete
   static void delete_argoanacLcLStage1Efficiency(void *p) {
      delete ((::argoana::Stage1Efficiency*)p);
   }
   static void deleteArray_argoanacLcLStage1Efficiency(void *p) {
      delete [] ((::argoana::Stage1Efficiency*)p);
   }
   static void destruct_argoanacLcLStage1Efficiency(void *p) {
      typedef ::argoana::Stage1Efficiency current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::argoana::Stage1Efficiency

namespace {
  void TriggerDictionaryInitialization_libargoneut_electrons_nuexsecanalysis_Impl() {
    static const char* headers[] = {
"Stage1Efficiency.h",
0
    };
    static const char* includePaths[] = {
"/data/products/larsoft/gallery/v1_03_08/include",
"/data/products/larsoft/canvas/v1_05_01/include",
"/data/products/larsoft/cetlib/v1_21_00/include",
"/data/products/larsoft/fhiclcpp/v4_02_00/include",
"/data/products/larsoft/lardataobj/v1_08_00/include",
"/data/products/larsoft/nusimdata/v1_06_01/include",
"/data/products/larsoft/larcoreobj/v1_06_01/include",
"/home/cadams/gallery_software/core",
"/argoneut_electrons",
"/data/products/larsoft/root/v6_06_08/Linux64bit+4.4-2.23-e10-nu-prof/include",
"/home/cadams/gallery_software/UserDev/nue_xsec/",
0
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "libargoneut_electrons_nuexsecanalysis dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_Autoloading_Map;
namespace argoana{class __attribute__((annotate("$clingAutoload$Stage1Efficiency.h")))  Stage1Efficiency;}
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "libargoneut_electrons_nuexsecanalysis dictionary payload"

#ifndef G__VECTOR_HAS_CLASS_ITERATOR
  #define G__VECTOR_HAS_CLASS_ITERATOR 1
#endif

#define _BACKWARD_BACKWARD_WARNING_H
#include "Stage1Efficiency.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[]={
"argoana::Stage1Efficiency", payloadCode, "@",
nullptr};

    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("libargoneut_electrons_nuexsecanalysis",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_libargoneut_electrons_nuexsecanalysis_Impl, {}, classesHeaders);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_libargoneut_electrons_nuexsecanalysis_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_libargoneut_electrons_nuexsecanalysis() {
  TriggerDictionaryInitialization_libargoneut_electrons_nuexsecanalysis_Impl();
}
