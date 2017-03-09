// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME gallery_framework_BaseDict

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
#include "DetectorStatusTypes.h"
#include "AnalysisConstants.h"
#include "MCConstants.h"
#include "GeoTypes.h"
#include "RawConstants.h"
#include "GeoConstants.h"
#include "messenger.h"
#include "FrameworkConstants.h"

// Header files passed via #pragma extra_include

namespace galleryfmwk {
   namespace ROOT {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *galleryfmwk_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("galleryfmwk", 0 /*version*/, "DetectorStatusTypes.h", 4,
                     ::ROOT::Internal::DefineBehavior((void*)0,(void*)0),
                     &galleryfmwk_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *galleryfmwk_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}

namespace galleryfmwk {
   namespace larch {
   namespace ROOT {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *galleryfmwkcLcLlarch_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("galleryfmwk::larch", 0 /*version*/, "DetectorStatusTypes.h", 5,
                     ::ROOT::Internal::DefineBehavior((void*)0,(void*)0),
                     &galleryfmwkcLcLlarch_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *galleryfmwkcLcLlarch_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}
}

namespace galleryfmwk {
   namespace anab {
   namespace ROOT {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *galleryfmwkcLcLanab_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("galleryfmwk::anab", 0 /*version*/, "AnalysisConstants.h", 6,
                     ::ROOT::Internal::DefineBehavior((void*)0,(void*)0),
                     &galleryfmwkcLcLanab_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *galleryfmwkcLcLanab_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}
}

namespace galleryfmwk {
   namespace simb {
   namespace ROOT {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *galleryfmwkcLcLsimb_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("galleryfmwk::simb", 0 /*version*/, "MCConstants.h", 23,
                     ::ROOT::Internal::DefineBehavior((void*)0,(void*)0),
                     &galleryfmwkcLcLsimb_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *galleryfmwkcLcLsimb_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}
}

namespace galleryfmwk {
   namespace geo {
   namespace ROOT {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *galleryfmwkcLcLgeo_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("galleryfmwk::geo", 0 /*version*/, "GeoTypes.h", 23,
                     ::ROOT::Internal::DefineBehavior((void*)0,(void*)0),
                     &galleryfmwkcLcLgeo_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *galleryfmwkcLcLgeo_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}
}

namespace galleryfmwk {
   namespace data {
   namespace ROOT {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *galleryfmwkcLcLdata_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("galleryfmwk::data", 0 /*version*/, "FrameworkConstants.h", 23,
                     ::ROOT::Internal::DefineBehavior((void*)0,(void*)0),
                     &galleryfmwkcLcLdata_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *galleryfmwkcLcLdata_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}
}

namespace galleryfmwk {
   namespace msg {
   namespace ROOT {
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();
      static TClass *galleryfmwkcLcLmsg_Dictionary();

      // Function generating the singleton type initializer
      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()
      {
         static ::ROOT::TGenericClassInfo 
            instance("galleryfmwk::msg", 0 /*version*/, "FrameworkConstants.h", 37,
                     ::ROOT::Internal::DefineBehavior((void*)0,(void*)0),
                     &galleryfmwkcLcLmsg_Dictionary, 0);
         return &instance;
      }
      // Insure that the inline function is _not_ optimized away by the compiler
      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_(InitFunctionKeeper))() = &GenerateInitInstance;  
      // Static variable to force the class initialization
      static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstance(); R__UseDummy(_R__UNIQUE_(Init));

      // Dictionary for non-ClassDef classes
      static TClass *galleryfmwkcLcLmsg_Dictionary() {
         return GenerateInitInstance()->GetClass();
      }

   }
}
}

namespace ROOT {
   static TClass *galleryfmwkcLcLgeocLcLTPCID_Dictionary();
   static void galleryfmwkcLcLgeocLcLTPCID_TClassManip(TClass*);
   static void *new_galleryfmwkcLcLgeocLcLTPCID(void *p = 0);
   static void *newArray_galleryfmwkcLcLgeocLcLTPCID(Long_t size, void *p);
   static void delete_galleryfmwkcLcLgeocLcLTPCID(void *p);
   static void deleteArray_galleryfmwkcLcLgeocLcLTPCID(void *p);
   static void destruct_galleryfmwkcLcLgeocLcLTPCID(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::galleryfmwk::geo::TPCID*)
   {
      ::galleryfmwk::geo::TPCID *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::galleryfmwk::geo::TPCID));
      static ::ROOT::TGenericClassInfo 
         instance("galleryfmwk::geo::TPCID", "GeoTypes.h", 26,
                  typeid(::galleryfmwk::geo::TPCID), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &galleryfmwkcLcLgeocLcLTPCID_Dictionary, isa_proxy, 4,
                  sizeof(::galleryfmwk::geo::TPCID) );
      instance.SetNew(&new_galleryfmwkcLcLgeocLcLTPCID);
      instance.SetNewArray(&newArray_galleryfmwkcLcLgeocLcLTPCID);
      instance.SetDelete(&delete_galleryfmwkcLcLgeocLcLTPCID);
      instance.SetDeleteArray(&deleteArray_galleryfmwkcLcLgeocLcLTPCID);
      instance.SetDestructor(&destruct_galleryfmwkcLcLgeocLcLTPCID);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::galleryfmwk::geo::TPCID*)
   {
      return GenerateInitInstanceLocal((::galleryfmwk::geo::TPCID*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const ::galleryfmwk::geo::TPCID*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *galleryfmwkcLcLgeocLcLTPCID_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::galleryfmwk::geo::TPCID*)0x0)->GetClass();
      galleryfmwkcLcLgeocLcLTPCID_TClassManip(theClass);
   return theClass;
   }

   static void galleryfmwkcLcLgeocLcLTPCID_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *galleryfmwkcLcLgeocLcLPlaneID_Dictionary();
   static void galleryfmwkcLcLgeocLcLPlaneID_TClassManip(TClass*);
   static void *new_galleryfmwkcLcLgeocLcLPlaneID(void *p = 0);
   static void *newArray_galleryfmwkcLcLgeocLcLPlaneID(Long_t size, void *p);
   static void delete_galleryfmwkcLcLgeocLcLPlaneID(void *p);
   static void deleteArray_galleryfmwkcLcLgeocLcLPlaneID(void *p);
   static void destruct_galleryfmwkcLcLgeocLcLPlaneID(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::galleryfmwk::geo::PlaneID*)
   {
      ::galleryfmwk::geo::PlaneID *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::galleryfmwk::geo::PlaneID));
      static ::ROOT::TGenericClassInfo 
         instance("galleryfmwk::geo::PlaneID", "GeoTypes.h", 60,
                  typeid(::galleryfmwk::geo::PlaneID), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &galleryfmwkcLcLgeocLcLPlaneID_Dictionary, isa_proxy, 4,
                  sizeof(::galleryfmwk::geo::PlaneID) );
      instance.SetNew(&new_galleryfmwkcLcLgeocLcLPlaneID);
      instance.SetNewArray(&newArray_galleryfmwkcLcLgeocLcLPlaneID);
      instance.SetDelete(&delete_galleryfmwkcLcLgeocLcLPlaneID);
      instance.SetDeleteArray(&deleteArray_galleryfmwkcLcLgeocLcLPlaneID);
      instance.SetDestructor(&destruct_galleryfmwkcLcLgeocLcLPlaneID);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::galleryfmwk::geo::PlaneID*)
   {
      return GenerateInitInstanceLocal((::galleryfmwk::geo::PlaneID*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const ::galleryfmwk::geo::PlaneID*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *galleryfmwkcLcLgeocLcLPlaneID_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::galleryfmwk::geo::PlaneID*)0x0)->GetClass();
      galleryfmwkcLcLgeocLcLPlaneID_TClassManip(theClass);
   return theClass;
   }

   static void galleryfmwkcLcLgeocLcLPlaneID_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *galleryfmwkcLcLgeocLcLWireID_Dictionary();
   static void galleryfmwkcLcLgeocLcLWireID_TClassManip(TClass*);
   static void *new_galleryfmwkcLcLgeocLcLWireID(void *p = 0);
   static void *newArray_galleryfmwkcLcLgeocLcLWireID(Long_t size, void *p);
   static void delete_galleryfmwkcLcLgeocLcLWireID(void *p);
   static void deleteArray_galleryfmwkcLcLgeocLcLWireID(void *p);
   static void destruct_galleryfmwkcLcLgeocLcLWireID(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::galleryfmwk::geo::WireID*)
   {
      ::galleryfmwk::geo::WireID *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::galleryfmwk::geo::WireID));
      static ::ROOT::TGenericClassInfo 
         instance("galleryfmwk::geo::WireID", "GeoTypes.h", 107,
                  typeid(::galleryfmwk::geo::WireID), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &galleryfmwkcLcLgeocLcLWireID_Dictionary, isa_proxy, 4,
                  sizeof(::galleryfmwk::geo::WireID) );
      instance.SetNew(&new_galleryfmwkcLcLgeocLcLWireID);
      instance.SetNewArray(&newArray_galleryfmwkcLcLgeocLcLWireID);
      instance.SetDelete(&delete_galleryfmwkcLcLgeocLcLWireID);
      instance.SetDeleteArray(&deleteArray_galleryfmwkcLcLgeocLcLWireID);
      instance.SetDestructor(&destruct_galleryfmwkcLcLgeocLcLWireID);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::galleryfmwk::geo::WireID*)
   {
      return GenerateInitInstanceLocal((::galleryfmwk::geo::WireID*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const ::galleryfmwk::geo::WireID*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *galleryfmwkcLcLgeocLcLWireID_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::galleryfmwk::geo::WireID*)0x0)->GetClass();
      galleryfmwkcLcLgeocLcLWireID_TClassManip(theClass);
   return theClass;
   }

   static void galleryfmwkcLcLgeocLcLWireID_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   static TClass *galleryfmwkcLcLMessage_Dictionary();
   static void galleryfmwkcLcLMessage_TClassManip(TClass*);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::galleryfmwk::Message*)
   {
      ::galleryfmwk::Message *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::galleryfmwk::Message));
      static ::ROOT::TGenericClassInfo 
         instance("galleryfmwk::Message", "messenger.h", 28,
                  typeid(::galleryfmwk::Message), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &galleryfmwkcLcLMessage_Dictionary, isa_proxy, 4,
                  sizeof(::galleryfmwk::Message) );
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::galleryfmwk::Message*)
   {
      return GenerateInitInstanceLocal((::galleryfmwk::Message*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const ::galleryfmwk::Message*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *galleryfmwkcLcLMessage_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::galleryfmwk::Message*)0x0)->GetClass();
      galleryfmwkcLcLMessage_TClassManip(theClass);
   return theClass;
   }

   static void galleryfmwkcLcLMessage_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_galleryfmwkcLcLgeocLcLTPCID(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::galleryfmwk::geo::TPCID : new ::galleryfmwk::geo::TPCID;
   }
   static void *newArray_galleryfmwkcLcLgeocLcLTPCID(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::galleryfmwk::geo::TPCID[nElements] : new ::galleryfmwk::geo::TPCID[nElements];
   }
   // Wrapper around operator delete
   static void delete_galleryfmwkcLcLgeocLcLTPCID(void *p) {
      delete ((::galleryfmwk::geo::TPCID*)p);
   }
   static void deleteArray_galleryfmwkcLcLgeocLcLTPCID(void *p) {
      delete [] ((::galleryfmwk::geo::TPCID*)p);
   }
   static void destruct_galleryfmwkcLcLgeocLcLTPCID(void *p) {
      typedef ::galleryfmwk::geo::TPCID current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::galleryfmwk::geo::TPCID

namespace ROOT {
   // Wrappers around operator new
   static void *new_galleryfmwkcLcLgeocLcLPlaneID(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::galleryfmwk::geo::PlaneID : new ::galleryfmwk::geo::PlaneID;
   }
   static void *newArray_galleryfmwkcLcLgeocLcLPlaneID(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::galleryfmwk::geo::PlaneID[nElements] : new ::galleryfmwk::geo::PlaneID[nElements];
   }
   // Wrapper around operator delete
   static void delete_galleryfmwkcLcLgeocLcLPlaneID(void *p) {
      delete ((::galleryfmwk::geo::PlaneID*)p);
   }
   static void deleteArray_galleryfmwkcLcLgeocLcLPlaneID(void *p) {
      delete [] ((::galleryfmwk::geo::PlaneID*)p);
   }
   static void destruct_galleryfmwkcLcLgeocLcLPlaneID(void *p) {
      typedef ::galleryfmwk::geo::PlaneID current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::galleryfmwk::geo::PlaneID

namespace ROOT {
   // Wrappers around operator new
   static void *new_galleryfmwkcLcLgeocLcLWireID(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::galleryfmwk::geo::WireID : new ::galleryfmwk::geo::WireID;
   }
   static void *newArray_galleryfmwkcLcLgeocLcLWireID(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) ::galleryfmwk::geo::WireID[nElements] : new ::galleryfmwk::geo::WireID[nElements];
   }
   // Wrapper around operator delete
   static void delete_galleryfmwkcLcLgeocLcLWireID(void *p) {
      delete ((::galleryfmwk::geo::WireID*)p);
   }
   static void deleteArray_galleryfmwkcLcLgeocLcLWireID(void *p) {
      delete [] ((::galleryfmwk::geo::WireID*)p);
   }
   static void destruct_galleryfmwkcLcLgeocLcLWireID(void *p) {
      typedef ::galleryfmwk::geo::WireID current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::galleryfmwk::geo::WireID

namespace ROOT {
} // end of namespace ROOT for class ::galleryfmwk::Message

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
                  &vectorlEstringgR_Dictionary, isa_proxy, 4,
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
   static TClass *vectorlEgalleryfmwkcLcLgeocLcLView_tgR_Dictionary();
   static void vectorlEgalleryfmwkcLcLgeocLcLView_tgR_TClassManip(TClass*);
   static void *new_vectorlEgalleryfmwkcLcLgeocLcLView_tgR(void *p = 0);
   static void *newArray_vectorlEgalleryfmwkcLcLgeocLcLView_tgR(Long_t size, void *p);
   static void delete_vectorlEgalleryfmwkcLcLgeocLcLView_tgR(void *p);
   static void deleteArray_vectorlEgalleryfmwkcLcLgeocLcLView_tgR(void *p);
   static void destruct_vectorlEgalleryfmwkcLcLgeocLcLView_tgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<galleryfmwk::geo::View_t>*)
   {
      vector<galleryfmwk::geo::View_t> *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<galleryfmwk::geo::View_t>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<galleryfmwk::geo::View_t>", -2, "vector", 214,
                  typeid(vector<galleryfmwk::geo::View_t>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlEgalleryfmwkcLcLgeocLcLView_tgR_Dictionary, isa_proxy, 4,
                  sizeof(vector<galleryfmwk::geo::View_t>) );
      instance.SetNew(&new_vectorlEgalleryfmwkcLcLgeocLcLView_tgR);
      instance.SetNewArray(&newArray_vectorlEgalleryfmwkcLcLgeocLcLView_tgR);
      instance.SetDelete(&delete_vectorlEgalleryfmwkcLcLgeocLcLView_tgR);
      instance.SetDeleteArray(&deleteArray_vectorlEgalleryfmwkcLcLgeocLcLView_tgR);
      instance.SetDestructor(&destruct_vectorlEgalleryfmwkcLcLgeocLcLView_tgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<galleryfmwk::geo::View_t> >()));
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const vector<galleryfmwk::geo::View_t>*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEgalleryfmwkcLcLgeocLcLView_tgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<galleryfmwk::geo::View_t>*)0x0)->GetClass();
      vectorlEgalleryfmwkcLcLgeocLcLView_tgR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlEgalleryfmwkcLcLgeocLcLView_tgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlEgalleryfmwkcLcLgeocLcLView_tgR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<galleryfmwk::geo::View_t> : new vector<galleryfmwk::geo::View_t>;
   }
   static void *newArray_vectorlEgalleryfmwkcLcLgeocLcLView_tgR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<galleryfmwk::geo::View_t>[nElements] : new vector<galleryfmwk::geo::View_t>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlEgalleryfmwkcLcLgeocLcLView_tgR(void *p) {
      delete ((vector<galleryfmwk::geo::View_t>*)p);
   }
   static void deleteArray_vectorlEgalleryfmwkcLcLgeocLcLView_tgR(void *p) {
      delete [] ((vector<galleryfmwk::geo::View_t>*)p);
   }
   static void destruct_vectorlEgalleryfmwkcLcLgeocLcLView_tgR(void *p) {
      typedef vector<galleryfmwk::geo::View_t> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<galleryfmwk::geo::View_t>

namespace ROOT {
   static TClass *vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR_Dictionary();
   static void vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR_TClassManip(TClass*);
   static void *new_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR(void *p = 0);
   static void *newArray_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR(Long_t size, void *p);
   static void delete_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR(void *p);
   static void deleteArray_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR(void *p);
   static void destruct_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const vector<galleryfmwk::geo::SigType_t>*)
   {
      vector<galleryfmwk::geo::SigType_t> *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(vector<galleryfmwk::geo::SigType_t>));
      static ::ROOT::TGenericClassInfo 
         instance("vector<galleryfmwk::geo::SigType_t>", -2, "vector", 214,
                  typeid(vector<galleryfmwk::geo::SigType_t>), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR_Dictionary, isa_proxy, 4,
                  sizeof(vector<galleryfmwk::geo::SigType_t>) );
      instance.SetNew(&new_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR);
      instance.SetNewArray(&newArray_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR);
      instance.SetDelete(&delete_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR);
      instance.SetDeleteArray(&deleteArray_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR);
      instance.SetDestructor(&destruct_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR);
      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<galleryfmwk::geo::SigType_t> >()));
      return &instance;
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const vector<galleryfmwk::geo::SigType_t>*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const vector<galleryfmwk::geo::SigType_t>*)0x0)->GetClass();
      vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR_TClassManip(theClass);
   return theClass;
   }

   static void vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR(void *p) {
      return  p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<galleryfmwk::geo::SigType_t> : new vector<galleryfmwk::geo::SigType_t>;
   }
   static void *newArray_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR(Long_t nElements, void *p) {
      return p ? ::new((::ROOT::Internal::TOperatorNewHelper*)p) vector<galleryfmwk::geo::SigType_t>[nElements] : new vector<galleryfmwk::geo::SigType_t>[nElements];
   }
   // Wrapper around operator delete
   static void delete_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR(void *p) {
      delete ((vector<galleryfmwk::geo::SigType_t>*)p);
   }
   static void deleteArray_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR(void *p) {
      delete [] ((vector<galleryfmwk::geo::SigType_t>*)p);
   }
   static void destruct_vectorlEgalleryfmwkcLcLgeocLcLSigType_tgR(void *p) {
      typedef vector<galleryfmwk::geo::SigType_t> current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class vector<galleryfmwk::geo::SigType_t>

namespace {
  void TriggerDictionaryInitialization_libgallery_framework_Base_Impl() {
    static const char* headers[] = {
"DetectorStatusTypes.h",
"AnalysisConstants.h",
"MCConstants.h",
"GeoTypes.h",
"RawConstants.h",
"GeoConstants.h",
"messenger.h",
"FrameworkConstants.h",
0
    };
    static const char* includePaths[] = {
"/data/products/larsoft/root/v6_06_08/Linux64bit+4.4-2.23-e10-nu-prof/include",
"/home/cadams/gallery_software/core/Base/",
0
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "libgallery_framework_Base dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_Autoloading_Map;
namespace galleryfmwk{namespace geo{struct __attribute__((annotate("$clingAutoload$GeoTypes.h")))  TPCID;}}
namespace galleryfmwk{namespace geo{struct __attribute__((annotate("$clingAutoload$GeoTypes.h")))  PlaneID;}}
namespace galleryfmwk{namespace geo{struct __attribute__((annotate("$clingAutoload$GeoTypes.h")))  WireID;}}
namespace galleryfmwk{namespace geo{enum  __attribute__((annotate("$clingAutoload$GeoConstants.h"))) SigType_t : unsigned int;}}
namespace std{template <typename _Tp> class __attribute__((annotate("$clingAutoload$string")))  allocator;
}
namespace galleryfmwk{namespace geo{enum  __attribute__((annotate("$clingAutoload$GeoConstants.h"))) View_t : unsigned int;}}
namespace std{template <class _CharT> struct __attribute__((annotate("$clingAutoload$string")))  char_traits;
}
namespace galleryfmwk{class __attribute__((annotate("$clingAutoload$messenger.h")))  Message;}
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "libgallery_framework_Base dictionary payload"

#ifndef G__VECTOR_HAS_CLASS_ITERATOR
  #define G__VECTOR_HAS_CLASS_ITERATOR 1
#endif

#define _BACKWARD_BACKWARD_WARNING_H
#include "DetectorStatusTypes.h"
#include "AnalysisConstants.h"
#include "MCConstants.h"
#include "GeoTypes.h"
#include "RawConstants.h"
#include "GeoConstants.h"
#include "messenger.h"
#include "FrameworkConstants.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[]={
"galleryfmwk::Message", payloadCode, "@",
"galleryfmwk::geo::PlaneID", payloadCode, "@",
"galleryfmwk::geo::TPCID", payloadCode, "@",
"galleryfmwk::geo::WireID", payloadCode, "@",
nullptr};

    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("libgallery_framework_Base",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_libgallery_framework_Base_Impl, {}, classesHeaders);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_libgallery_framework_Base_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_libgallery_framework_Base() {
  TriggerDictionaryInitialization_libgallery_framework_Base_Impl();
}
