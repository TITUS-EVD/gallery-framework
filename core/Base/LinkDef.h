//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace galleryfmwk+;
#pragma link C++ namespace galleryfmwk::simb+;
#pragma link C++ namespace galleryfmwk::anab+;
#pragma link C++ namespace galleryfmwk::msg+;
#pragma link C++ namespace galleryfmwk::larch+;
#pragma link C++ namespace galleryfmwk::data+;
// #pragma link C++ class galleryfmwk::data+;
#pragma link C++ namespace galleryfmwk::geo+;

#pragma link C++ enum galleryfmwk::geo::SigType_t+;
#pragma link C++ enum galleryfmwk::geo::View_t+;

#pragma link C++ struct galleryfmwk::geo::TPCID+;
#pragma link C++ struct galleryfmwk::geo::PlaneID+;
#pragma link C++ struct galleryfmwk::geo::WireID+;

#pragma link C++ class std::vector<galleryfmwk::geo::SigType_t>+;
#pragma link C++ class std::vector<galleryfmwk::geo::View_t>+;
#pragma link C++ class std::vector<std::string>+;
#pragma link C++ enum galleryfmwk::data::DataType_t+;

#pragma link C++ class galleryfmwk::Message+;
#pragma link C++ class galleryfmwk::larlite_base+;

#pragma link C++ function const std::string& GetProductName(const galleryfmwk::data::DataType_t)+;

#endif
