// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME gallery_framework_AnalysisDict

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
#include "ana_base.h"
#include "ana_processor.h"

// Header files passed via #pragma extra_include

namespace ROOT {
   static TClass *galleryfmwkcLcLana_base_Dictionary();
   static void galleryfmwkcLcLana_base_TClassManip(TClass*);
   static void *new_galleryfmwkcLcLana_base(void *p = 0);
   static void *newArray_galleryfmwkcLcLana_base(Long_t size, void *p);
   static void delete_galleryfmwkcLcLana_base(void *p);
   static void deleteArray_galleryfmwkcLcLana_base(void *p);
   static void destruct_galleryfmwkcLcLana_base(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::galleryfmwk::ana_base*)
   {
      ::galleryfmwk::ana_base *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::galleryfmwk::ana_base));
      static ::ROOT::TGenericClassInfo 
         instance("galleryfmwk::ana_base", "ana_base.h", 29,
                  typeid(::galleryfmwk::ana_base), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &galleryfmwkcLcLana_base_Dictionary, isa_proxy, 4,
                  sizeof(::galleryfmwk::ana_base) );
      instance.SetNew(&new_galleryfmwkcLcLana_base);
      instance.SetNewArray(&newArray_galleryfmwkcLcLana_base);
      instance.SetDelete(&delete_galleryfmwkcLcLana_base);
      instance.SetDeleteArray(&deleteArray_galleryfmwkcLcLana_base);
      instance.SetDestructor(&destruct_galleryfmwkcLcLana_base);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::galleryfmwk::ana_base*)
   {
      return GenerateInitInstanceLocal((::galleryfmwk::ana_base*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const ::galleryfmwk::ana_base*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *galleryfmwkcLcLana_base_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::galleryfmwk::ana_base*)0x0)->GetClass();
      galleryfmwkcLcLana_base_TClassManip(theClass);
   return theClass;
   }

   static void galleryfmwkcLcLana_base_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *galleryfmwkcLcLana_processor_Dictionary();
   static void galleryfmwkcLcLana_processor_TClassManip(TClass*);
   static void *new_galleryfmwkcLcLana_processor(void *p = 0);
   static void *newArray_galleryfmwkcLcLana_processor(Long_t size, void *p);
   static void delete_galleryfmwkcLcLana_processor(void *p);
   static void deleteArray_galleryfmwkcLcLana_processor(void *p);
   static void destruct_galleryfmwkcLcLana_processor(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::galleryfmwk::ana_processor*)
   {
      ::galleryfmwk::ana_processor *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::galleryfmwk::ana_processor));
      static ::ROOT::TGenericClassInfo 
         instance("galleryfmwk::ana_processor", "ana_processor.h", 28,
                  typeid(::galleryfmwk::ana_processor), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &galleryfmwkcLcLana_processor_Dictionary, isa_proxy, 4,
                  sizeof(::galleryfmwk::ana_processor) );
      instance.SetNew(&new_galleryfmwkcLcLana_processor);
      instance.SetNewArray(&newArray_galleryfmwkcLcLana_processor);
      instance.SetDelete(&delete_galleryfmwkcLcLana_processor);
      instance.SetDeleteArray(&deleteArray_galleryfmwkcLcLana_processor);
      instance.SetDestructor(&destruct_galleryfmwkcLcLana_processor);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::galleryfmwk::ana_processor*)
   {
      return GenerateInitInstanceLocal((::galleryfmwk::ana_processor*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const ::galleryfmwk::ana_processor*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *galleryfmwkcLcLana_processor_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::galleryfmwk::ana_processor*)0x0)->GetClass();
      galleryfmwkcLcLana_processor_TClassManip(theClass);
   return theClass;
   }

   static void galleryfmwkcLcLana_processor_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_galleryfmwkcLcLana_base(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::galleryfmwk::ana_base : new ::galleryfmwk::ana_base;
   }
   static void *newArray_galleryfmwkcLcLana_base(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::galleryfmwk::ana_base[nElements] : new ::galleryfmwk::ana_base[nElements];
   }
   // Wrapper around operator delete
   static void delete_galleryfmwkcLcLana_base(void *p) {
      delete ((::galleryfmwk::ana_base*)p);
   }
   static void deleteArray_galleryfmwkcLcLana_base(void *p) {
      delete [] ((::galleryfmwk::ana_base*)p);
   }
   static void destruct_galleryfmwkcLcLana_base(void *p) {
      typedef ::galleryfmwk::ana_base current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::galleryfmwk::ana_base

namespace ROOT {
   // Wrappers around operator new
   static void *new_galleryfmwkcLcLana_processor(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::galleryfmwk::ana_processor : new ::galleryfmwk::ana_processor;
   }
   static void *newArray_galleryfmwkcLcLana_processor(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::galleryfmwk::ana_processor[nElements] : new ::galleryfmwk::ana_processor[nElements];
   }
   // Wrapper around operator delete
   static void delete_galleryfmwkcLcLana_processor(void *p) {
      delete ((::galleryfmwk::ana_processor*)p);
   }
   static void deleteArray_galleryfmwkcLcLana_processor(void *p) {
      delete [] ((::galleryfmwk::ana_processor*)p);
   }
   static void destruct_galleryfmwkcLcLana_processor(void *p) {
      typedef ::galleryfmwk::ana_processor current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::galleryfmwk::ana_processor

namespace ROOT {
   static TClass *vectorlEstringgR_Dictionary();
   static void vectorlEstringgR_TClassManip(TClass*);
   static void *new_vectorlEstringgR(void *p = 0);
   static void *newArray_vectorlEstringgR(Long_t size, void *p);
   static void delete_vectorlEstringgR(void *p);
   static void deleteArray_vectorlEstringgR(void *p);
   static void destruct_vectorlEstringgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<string>*)
   {
      vector<string> *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<string>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<string>", -2, "vector", 214,
                  typeid(vector<string>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlEstringgR_Dictionary, isa_proxy, 0,
                  sizeof(vector<string>) );
      instance.SetNew(&new_vectorlEstringgR);
      instance.SetNewArray(&newArray_vectorlEstringgR);
      instance.SetDelete(&delete_vectorlEstringgR);
      instance.SetDeleteArray(&deleteArray_vectorlEstringgR);
      instance.SetDestructor(&destruct_vectorlEstringgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<string> >()));
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const vector<string>*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEstringgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<string>*)0x0)->GetClass();
      vectorlEstringgR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlEstringgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlEstringgR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<string> : new vector<string>;
   }
   static void *newArray_vectorlEstringgR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<string>[nElements] : new vector<string>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlEstringgR(void *p) {
      delete ((vector<string>*)p);
   }
   static void deleteArray_vectorlEstringgR(void *p) {
      delete [] ((vector<string>*)p);
   }
   static void destruct_vectorlEstringgR(void *p) {
      typedef vector<string> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<string>

namespace ROOT {
   static TClass *vectorlEgalleryfmwkcLcLana_basemUgR_Dictionary();
   static void vectorlEgalleryfmwkcLcLana_basemUgR_TClassManip(TClass*);
   static void *new_vectorlEgalleryfmwkcLcLana_basemUgR(void *p = 0);
   static void *newArray_vectorlEgalleryfmwkcLcLana_basemUgR(Long_t size, void *p);
   static void delete_vectorlEgalleryfmwkcLcLana_basemUgR(void *p);
   static void deleteArray_vectorlEgalleryfmwkcLcLana_basemUgR(void *p);
   static void destruct_vectorlEgalleryfmwkcLcLana_basemUgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<galleryfmwk::ana_base*>*)
   {
      vector<galleryfmwk::ana_base*> *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<galleryfmwk::ana_base*>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<galleryfmwk::ana_base*>", -2, "vector", 214,
                  typeid(vector<galleryfmwk::ana_base*>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlEgalleryfmwkcLcLana_basemUgR_Dictionary, isa_proxy, 4,
                  sizeof(vector<galleryfmwk::ana_base*>) );
      instance.SetNew(&new_vectorlEgalleryfmwkcLcLana_basemUgR);
      instance.SetNewArray(&newArray_vectorlEgalleryfmwkcLcLana_basemUgR);
      instance.SetDelete(&delete_vectorlEgalleryfmwkcLcLana_basemUgR);
      instance.SetDeleteArray(&deleteArray_vectorlEgalleryfmwkcLcLana_basemUgR);
      instance.SetDestructor(&destruct_vectorlEgalleryfmwkcLcLana_basemUgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<galleryfmwk::ana_base*> >()));
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const vector<galleryfmwk::ana_base*>*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEgalleryfmwkcLcLana_basemUgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<galleryfmwk::ana_base*>*)0x0)->GetClass();
      vectorlEgalleryfmwkcLcLana_basemUgR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlEgalleryfmwkcLcLana_basemUgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlEgalleryfmwkcLcLana_basemUgR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<galleryfmwk::ana_base*> : new vector<galleryfmwk::ana_base*>;
   }
   static void *newArray_vectorlEgalleryfmwkcLcLana_basemUgR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<galleryfmwk::ana_base*>[nElements] : new vector<galleryfmwk::ana_base*>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlEgalleryfmwkcLcLana_basemUgR(void *p) {
      delete ((vector<galleryfmwk::ana_base*>*)p);
   }
   static void deleteArray_vectorlEgalleryfmwkcLcLana_basemUgR(void *p) {
      delete [] ((vector<galleryfmwk::ana_base*>*)p);
   }
   static void destruct_vectorlEgalleryfmwkcLcLana_basemUgR(void *p) {
      typedef vector<galleryfmwk::ana_base*> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<galleryfmwk::ana_base*>

namespace ROOT {
   static TClass *vectorlEboolgR_Dictionary();
   static void vectorlEboolgR_TClassManip(TClass*);
   static void *new_vectorlEboolgR(void *p = 0);
   static void *newArray_vectorlEboolgR(Long_t size, void *p);
   static void delete_vectorlEboolgR(void *p);
   static void deleteArray_vectorlEboolgR(void *p);
   static void destruct_vectorlEboolgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<bool>*)
   {
      vector<bool> *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<bool>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<bool>", -2, "vector", 526,
                  typeid(vector<bool>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlEboolgR_Dictionary, isa_proxy, 0,
                  sizeof(vector<bool>) );
      instance.SetNew(&new_vectorlEboolgR);
      instance.SetNewArray(&newArray_vectorlEboolgR);
      instance.SetDelete(&delete_vectorlEboolgR);
      instance.SetDeleteArray(&deleteArray_vectorlEboolgR);
      instance.SetDestructor(&destruct_vectorlEboolgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<bool> >()));
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const vector<bool>*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEboolgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<bool>*)0x0)->GetClass();
      vectorlEboolgR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlEboolgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlEboolgR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<bool> : new vector<bool>;
   }
   static void *newArray_vectorlEboolgR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<bool>[nElements] : new vector<bool>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlEboolgR(void *p) {
      delete ((vector<bool>*)p);
   }
   static void deleteArray_vectorlEboolgR(void *p) {
      delete [] ((vector<bool>*)p);
   }
   static void destruct_vectorlEboolgR(void *p) {
      typedef vector<bool> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<bool>

namespace ROOT {
   static TClass *maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR_Dictionary();
   static void maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR_TClassManip(TClass*);
   static void *new_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR(void *p = 0);
   static void *newArray_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR(Long_t size, void *p);
   static void delete_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR(void *p);
   static void deleteArray_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR(void *p);
   static void destruct_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const map<galleryfmwk::ana_base*,unsigned long>*)
   {
      map<galleryfmwk::ana_base*,unsigned long> *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(map<galleryfmwk::ana_base*,unsigned long>));
      static ::ROOT::TGenericClassInfo 
         instance("map<galleryfmwk::ana_base*,unsigned long>", -2, "map", 96,
                  typeid(map<galleryfmwk::ana_base*,unsigned long>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR_Dictionary, isa_proxy, 0,
                  sizeof(map<galleryfmwk::ana_base*,unsigned long>) );
      instance.SetNew(&new_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR);
      instance.SetNewArray(&newArray_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR);
      instance.SetDelete(&delete_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR);
      instance.SetDeleteArray(&deleteArray_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR);
      instance.SetDestructor(&destruct_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::MapInsert< map<galleryfmwk::ana_base*,unsigned long> >()));
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const map<galleryfmwk::ana_base*,unsigned long>*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const map<galleryfmwk::ana_base*,unsigned long>*)0x0)->GetClass();
      maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR_TClassManip(theClass);
   return theClass;
   }

   static void maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) map<galleryfmwk::ana_base*,unsigned long> : new map<galleryfmwk::ana_base*,unsigned long>;
   }
   static void *newArray_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) map<galleryfmwk::ana_base*,unsigned long>[nElements] : new map<galleryfmwk::ana_base*,unsigned long>[nElements];
   }
   // Wrapper around operator delete
   static void delete_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR(void *p) {
      delete ((map<galleryfmwk::ana_base*,unsigned long>*)p);
   }
   static void deleteArray_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR(void *p) {
      delete [] ((map<galleryfmwk::ana_base*,unsigned long>*)p);
   }
   static void destruct_maplEgalleryfmwkcLcLana_basemUcOunsignedsPlonggR(void *p) {
      typedef map<galleryfmwk::ana_base*,unsigned long> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class map<galleryfmwk::ana_base*,unsigned long>

namespace {
  void TriggerDictionaryInitialization_libgallery_framework_Analysis_Impl() {
    static const char* headers[] = {
"ana_base.h",
"ana_processor.h",
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
"/data/products/larsoft/root/v6_06_08/Linux64bit+4.4-2.23-e10-nu-prof/include",
"/home/cadams/gallery_software/core/Analysis/",
0
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "libgallery_framework_Analysis dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_Autoloading_Map;
namespace galleryfmwk{class __attribute__((annotate("$clingAutoload$ana_base.h")))  ana_base;}
namespace std{template <typename _Tp> class __attribute__((annotate("$clingAutoload$string")))  allocator;
}
namespace galleryfmwk{class __attribute__((annotate("$clingAutoload$ana_processor.h")))  ana_processor;}
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "libgallery_framework_Analysis dictionary payload"

#ifndef G__VECTOR_HAS_CLASS_ITERATOR
  #define G__VECTOR_HAS_CLASS_ITERATOR 1
#endif

#define _BACKWARD_BACKWARD_WARNING_H
#include "ana_base.h"
#include "ana_processor.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[]={
"galleryfmwk::ana_base", payloadCode, "@",
"galleryfmwk::ana_processor", payloadCode, "@",
nullptr};

    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("libgallery_framework_Analysis",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_libgallery_framework_Analysis_Impl, {}, classesHeaders);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_libgallery_framework_Analysis_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_libgallery_framework_Analysis() {
  TriggerDictionaryInitialization_libgallery_framework_Analysis_Impl();
}
