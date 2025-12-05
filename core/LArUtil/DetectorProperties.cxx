#ifndef GALLERY_FMWK_DETECTORPROPERTIES_CXX
#define GALLERY_FMWK_DETECTORPROPERTIES_CXX

#include "DetectorProperties.h"

namespace larutil {

  DetectorProperties* DetectorProperties::_me = 0;

  DetectorProperties::DetectorProperties(bool default_load) : LArUtilBase()
  {
    if(default_load) {
      _file_name = Form("%s/LArUtil/dat/%s",
			getenv("GALLERY_FMWK_COREDIR"),
			kUTIL_DATA_FILENAME[LArUtilConfig::Detector()].c_str());
      _tree_name = kTREENAME_DETECTORPROPERTIES;
      LoadData();
      DumpInfo();
    } 
    // if(default_load) {
    //   std::cout << "************************** DetectorProperties" << std::endl;
    //   _file_name = Form("%s/LArUtil/dat/%s",
    //   getenv("GALLERY_FMWK_COREDIR"),
    //   kUTIL_FCL_FILENAME[LArUtilConfig::Detector()].c_str());
    //   _fcl_name = kTREENAME_DETECTORPROPERTIES;
    //   LoadDataFromServices();
    // }
  }

  void DetectorProperties::ClearData()
  {
    fSamplingRate = galleryfmwk::data::kINVALID_DOUBLE;
    fTriggerOffset = galleryfmwk::data::kINVALID_INT;
    fElectronsToADC = galleryfmwk::data::kINVALID_DOUBLE;
    fNumberTimeSamples = galleryfmwk::data::kINVALID_UINT;
    fReadOutWindowSize = galleryfmwk::data::kINVALID_UINT;

    fTimeOffsetU = galleryfmwk::data::kINVALID_DOUBLE;
    fTimeOffsetV = galleryfmwk::data::kINVALID_DOUBLE;
    fTimeOffsetZ = galleryfmwk::data::kINVALID_DOUBLE;

    fXTicksCoefficient = galleryfmwk::data::kINVALID_DOUBLE;
    fXTicksOffsets.clear();    

  }

  // bool DetectorProperties::ReadFromServices()
  // {

  //   ClearData();

  //   auto detp = LArUtilServicesHandler::GetDetProperties(_file_name);
  //   auto _geo_service = LArUtilServicesHandler::GetGeometry(_file_name);

  //   fSamplingRate = detp->SamplingRate();
  //   fTriggerOffset = detp->TriggerOffset();
  //   fElectronsToADC = detp->ElectronsToADC();
  //   fNumberTimeSamples = detp->NumberTimeSamples();
  //   fReadOutWindowSize = detp->ReadOutWindowSize();
  //   fTimeOffsetU = detp->TimeOffsetU();
  //   fTimeOffsetV = detp->TimeOffsetV();
  //   fTimeOffsetZ = detp->TimeOffsetZ();
  //   fXTicksCoefficient = detp->GetXTicksCoefficient(0, 0); 
    
  //   for (size_t tpc = 0; tpc < _geo_service->NTPC(); tpc++) {
  //     for (size_t plane = 0; plane < _geo_service->Nplanes(tpc); plane++) {
  //       fXTicksOffsets.push_back(detp->GetXTicksOffset(plane, tpc, 0));  
  //     }
  //   }
   
  //   return true;
  // }

  bool DetectorProperties::ReadTree()
  {

    ClearData();
    TChain *ch = new TChain(_tree_name.c_str());
    ch->AddFile(_file_name.c_str());

    std::string error_msg("");
    if(!(ch->GetBranch("fSamplingRate")))      error_msg += "      fSamplingRate\n";
    if(!(ch->GetBranch("fTriggerOffset")))     error_msg += "      fTriggerOffset\n";
    if(!(ch->GetBranch("fElectronsToADC")))    error_msg += "      fElectronsToADC\n";
    if(!(ch->GetBranch("fNumberTimeSamples"))) error_msg += "      fNumberTimeSamples\n";
    if(!(ch->GetBranch("fReadOutWindowSize"))) error_msg += "      fReadOutWindowSize\n";
    if(!(ch->GetBranch("fTimeOffsetU")))       error_msg += "      fTimeOffsetU\n";
    if(!(ch->GetBranch("fTimeOffsetV")))       error_msg += "      fTimeOffsetV\n";
    if(!(ch->GetBranch("fTimeOffsetZ")))       error_msg += "      fTimeOffsetZ\n";
    if(!(ch->GetBranch("fXTicksCoefficient"))) error_msg += "      fXTicksCoefficient\n";
    if(!(ch->GetBranch("fXTicksOffsets")))     error_msg += "      fXTicksOffsets\n";
    if(!error_msg.empty()) {

      throw LArUtilException(Form("Missing following TBranches...\n%s",error_msg.c_str()));

      return false;
    }

    ch->SetBranchAddress("fSamplingRate",&fSamplingRate);
    ch->SetBranchAddress("fTriggerOffset",&fTriggerOffset);
    ch->SetBranchAddress("fElectronsToADC",&fElectronsToADC);
    ch->SetBranchAddress("fNumberTimeSamples",&fNumberTimeSamples);
    ch->SetBranchAddress("fReadOutWindowSize",&fReadOutWindowSize);
    ch->SetBranchAddress("fTimeOffsetU",&fTimeOffsetU);
    ch->SetBranchAddress("fTimeOffsetV",&fTimeOffsetV);
    ch->SetBranchAddress("fTimeOffsetZ",&fTimeOffsetZ);
    ch->SetBranchAddress("fXTicksCoefficient",&fXTicksCoefficient);

    std::vector<Double_t> *pXTicksOffsets=nullptr;
    ch->SetBranchAddress("fXTicksOffsets",&pXTicksOffsets);

    ch->GetEntry(0);

    for(size_t i=0; i<pXTicksOffsets->size(); ++i)
      fXTicksOffsets.push_back(pXTicksOffsets->at(i));

    delete ch;
    return true;
  }

  void DetectorProperties::DumpInfo()
  {
    std::cout << std::endl;
    std::cout << "Dumping DetectorProperties info:" << std::endl;
    std::cout << "\tfSamplingRate: " << fSamplingRate << std::endl;
    std::cout << "\tfTriggerOffset: " << fTriggerOffset << std::endl;
    std::cout << "\tfElectronsToADC: " << fElectronsToADC << std::endl;
    std::cout << "\tfNumberTimeSamples: " << fNumberTimeSamples << std::endl;
    std::cout << "\tfReadOutWindowSize: " << fReadOutWindowSize << std::endl;
    std::cout << "\tfTimeOffsetU: " << fTimeOffsetU << std::endl;
    std::cout << "\tfTimeOffsetV: " << fTimeOffsetV << std::endl;
    std::cout << "\tfTimeOffsetZ: " << fTimeOffsetZ << std::endl;
    std::cout << "\tfXTicksCoefficient: " << fXTicksCoefficient << std::endl;
    std::cout << "\tfXTicksOffsets[0]: " << fXTicksOffsets[0] << std::endl;
    std::cout << std::endl;  

  }

}

#endif
