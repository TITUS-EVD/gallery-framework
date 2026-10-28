#include "ProductReader.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>

#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"
#include "gallery/Handle.h"
#include "lardataobj/RawData/RawDigit.h"
#include "lardataobj/RawData/raw.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/Wire.h"
#include "lardataobj/RecoBase/PFParticle.h"
#include "lardataobj/RecoBase/Shower.h"
#include "lardataobj/RecoBase/SpacePoint.h"
#include "lardataobj/RecoBase/Track.h"
#include "lardataobj/RecoBase/TrackHitMeta.h"
#include "lardataobj/RecoBase/Vertex.h"
#include "lardataobj/AnalysisBase/T0.h"
#include "canvas/Persistency/Common/Assns.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "canvas/Persistency/Provenance/ProductID.h"

#ifdef WITH_SBNOBJ
#include "sbnobj/ICARUS/TPC/ChannelROI.h"
#endif

namespace titusio {

  void HitColumns::clear()
  {
    channel.clear();
    cryostat.clear();
    tpc.clear();
    plane.clear();
    wire.clear();
    peak_time.clear();
    rms.clear();
    integral.clear();
    peak_amplitude.clear();
    start_tick.clear();
    end_tick.clear();
  }

  void WireColumns::clear()
  {
    channel.clear();
    n_ticks.clear();
    roi_row.clear();
    roi_start.clear();
    roi_offset.assign(1, 0);
    values.clear();
  }

  void RawDigitColumns::clear()
  {
    channel.clear();
    pedestal.clear();
    n_samples.clear();
    adc.clear();
    width = 0;
  }

  bool readHits(gallery::Event const& ev, std::string const& tag, HitColumns& out)
  {
    out.clear();
    gallery::Handle<std::vector<recob::Hit>> handle;
    if (!ev.getByLabel(art::InputTag(tag), handle)) return false;

    for (auto const& hit : *handle) {
      auto const& wid = hit.WireID();
      out.channel.push_back(hit.Channel());
      out.cryostat.push_back(wid.Cryostat);
      out.tpc.push_back(wid.TPC);
      out.plane.push_back(wid.Plane);
      out.wire.push_back(wid.Wire);
      out.peak_time.push_back(hit.PeakTime());
      out.rms.push_back(hit.RMS());
      out.integral.push_back(hit.Integral());
      out.peak_amplitude.push_back(hit.PeakAmplitude());
      out.start_tick.push_back(hit.StartTick());
      out.end_tick.push_back(hit.EndTick());
    }
    return true;
  }

  namespace {

    /// recob::Wire and recob::ChannelROI share Channel() and a
    /// lar::sparse_vector SignalROI()
    template <typename T>
    bool readSignals(gallery::Event const& ev, std::string const& tag, WireColumns& out)
    {
      out.clear();
      gallery::Handle<std::vector<T>> handle;
      if (!ev.getByLabel(art::InputTag(tag), handle)) return false;

      for (std::size_t row = 0; row < handle->size(); ++row) {
        auto const& signal = (*handle)[row].SignalROI();
        out.channel.push_back((*handle)[row].Channel());
        out.n_ticks.push_back(signal.size());
        for (auto const& range : signal.get_ranges()) {
          out.roi_row.push_back(row);
          out.roi_start.push_back(range.begin_index());
          out.values.insert(out.values.end(), range.begin(), range.end());
          out.roi_offset.push_back(out.values.size());
        }
      }
      return true;
    }

  } // namespace

  bool readWires(gallery::Event const& ev, std::string const& tag, WireColumns& out)
  {
    return readSignals<recob::Wire>(ev, tag, out);
  }

  bool readChannelROIs(gallery::Event const& ev, std::string const& tag, WireColumns& out)
  {
#ifdef WITH_SBNOBJ
    return readSignals<recob::ChannelROI>(ev, tag, out);
#else
    throw std::runtime_error("titusio was compiled without sbnobj: recob::ChannelROI is not available");
#endif
  }

  bool hasChannelROISupport()
  {
#ifdef WITH_SBNOBJ
    return true;
#else
    return false;
#endif
  }

  bool readRawDigits(gallery::Event const& ev, std::string const& tag, RawDigitColumns& out)
  {
    out.clear();
    gallery::Handle<std::vector<raw::RawDigit>> handle;
    if (!ev.getByLabel(art::InputTag(tag), handle)) return false;

    std::vector<std::vector<short>> rows;
    rows.reserve(handle->size());
    for (auto const& digit : *handle) {
      if (digit.Compression() == raw::kNone) {
        rows.push_back(digit.ADCs());
      }
      else {
        std::vector<short> samples(digit.Samples());
        raw::Uncompress(digit.ADCs(), samples, digit.Compression());
        rows.push_back(std::move(samples));
      }
      out.channel.push_back(digit.Channel());
      out.pedestal.push_back(digit.GetPedestal());
      out.n_samples.push_back(rows.back().size());
      out.width = std::max<unsigned long>(out.width, rows.back().size());
    }

    out.adc.assign(rows.size() * out.width, 0);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      std::copy(rows[row].begin(), rows[row].end(), out.adc.begin() + row * out.width);
    }
    return true;
  }

  void TrackColumns::clear()
  {
    id.clear();
    point_offset.assign(1, 0);
    x.clear();
    y.clear();
    z.clear();
    valid.clear();
  }

  void PointColumns::clear()
  {
    id.clear();
    x.clear();
    y.clear();
    z.clear();
  }

  void ShowerColumns::clear()
  {
    id.clear();
    start_x.clear();
    start_y.clear();
    start_z.clear();
    dir_x.clear();
    dir_y.clear();
    dir_z.clear();
    length.clear();
    open_angle.clear();
    best_plane.clear();
  }

  void PFParticleColumns::clear()
  {
    self_index.clear();
    pdg.clear();
    parent.clear();
  }

  void T0Columns::clear()
  {
    time.clear();
    trigger_type.clear();
    id.clear();
  }

  void AssociationColumns::clear()
  {
    left.clear();
    right.clear();
    left_product.clear();
    right_product.clear();
    meta_index.clear();
    product_tags.clear();
  }

  bool readTracks(gallery::Event const& ev, std::string const& tag, TrackColumns& out)
  {
    out.clear();
    gallery::Handle<std::vector<recob::Track>> handle;
    if (!ev.getByLabel(art::InputTag(tag), handle)) return false;

    for (auto const& track : *handle) {
      out.id.push_back(track.ID());
      for (std::size_t i = 0; i < track.NumberTrajectoryPoints(); ++i) {
        auto const& point = track.LocationAtPoint(i);
        out.x.push_back(point.X());
        out.y.push_back(point.Y());
        out.z.push_back(point.Z());
        out.valid.push_back(track.HasValidPoint(i) ? 1 : 0);
      }
      out.point_offset.push_back(out.x.size());
    }
    return true;
  }

  bool readVertices(gallery::Event const& ev, std::string const& tag, PointColumns& out)
  {
    out.clear();
    gallery::Handle<std::vector<recob::Vertex>> handle;
    if (!ev.getByLabel(art::InputTag(tag), handle)) return false;

    for (auto const& vertex : *handle) {
      auto const& position = vertex.position();
      out.id.push_back(vertex.ID());
      out.x.push_back(position.X());
      out.y.push_back(position.Y());
      out.z.push_back(position.Z());
    }
    return true;
  }

  bool readSpacePoints(gallery::Event const& ev, std::string const& tag, PointColumns& out)
  {
    out.clear();
    gallery::Handle<std::vector<recob::SpacePoint>> handle;
    if (!ev.getByLabel(art::InputTag(tag), handle)) return false;

    for (auto const& point : *handle) {
      auto const* xyz = point.XYZ();
      out.id.push_back(point.ID());
      out.x.push_back(xyz[0]);
      out.y.push_back(xyz[1]);
      out.z.push_back(xyz[2]);
    }
    return true;
  }

  bool readShowers(gallery::Event const& ev, std::string const& tag, ShowerColumns& out)
  {
    out.clear();
    gallery::Handle<std::vector<recob::Shower>> handle;
    if (!ev.getByLabel(art::InputTag(tag), handle)) return false;

    for (auto const& shower : *handle) {
      auto const& start = shower.ShowerStart();
      auto const& direction = shower.Direction();
      out.id.push_back(shower.ID());
      out.start_x.push_back(start.X());
      out.start_y.push_back(start.Y());
      out.start_z.push_back(start.Z());
      out.dir_x.push_back(direction.X());
      out.dir_y.push_back(direction.Y());
      out.dir_z.push_back(direction.Z());
      out.length.push_back(shower.Length());
      out.open_angle.push_back(shower.OpenAngle());
      out.best_plane.push_back(shower.best_plane());
    }
    return true;
  }

  bool readPFParticles(gallery::Event const& ev, std::string const& tag, PFParticleColumns& out)
  {
    out.clear();
    gallery::Handle<std::vector<recob::PFParticle>> handle;
    if (!ev.getByLabel(art::InputTag(tag), handle)) return false;

    for (auto const& pfp : *handle) {
      out.self_index.push_back(pfp.Self());
      out.pdg.push_back(pfp.PdgCode());
      out.parent.push_back(pfp.IsPrimary() ? -1 : static_cast<int>(pfp.Parent()));
    }
    return true;
  }

  bool readT0s(gallery::Event const& ev, std::string const& tag, T0Columns& out)
  {
    out.clear();
    gallery::Handle<std::vector<anab::T0>> handle;
    if (!ev.getByLabel(art::InputTag(tag), handle)) return false;

    for (auto const& t0 : *handle) {
      out.time.push_back(t0.Time());
      out.trigger_type.push_back(t0.TriggerType());
      out.id.push_back(t0.ID());
    }
    return true;
  }

  namespace {

    /// ProductID value of a product of type std::vector<T> with this tag, if any
    template <typename T>
    void addProductID(gallery::Event const& ev, std::string const& tag,
                      std::map<unsigned int, std::string>& ids)
    {
      try {
        auto const handle = ev.getValidHandle<std::vector<T>>(art::InputTag(tag));
        ids[handle.id().value()] = tag;
      }
      catch (...) {
        // not a std::vector<T> product in this event
      }
    }

    template <typename A, typename B, typename D>
    bool readAssns(gallery::Event const& ev, std::string const& tag, bool swapped,
                   std::vector<std::string> const& candidate_tags, AssociationColumns& out)
    {
      out.clear();
      gallery::Handle<art::Assns<A, B, D>> handle;
      if (!ev.getByLabel(art::InputTag(tag), handle)) return false;

      std::map<unsigned int, std::string> ids;
      for (auto const& candidate : candidate_tags) {
        addProductID<A>(ev, candidate, ids);
        addProductID<B>(ev, candidate, ids);
      }
      auto product_index = [&](art::ProductID const& id) {
        auto const found = ids.find(id.value());
        std::string const tag_of_id = (found == ids.end()) ? "" : found->second;
        auto const it = std::find(out.product_tags.begin(), out.product_tags.end(), tag_of_id);
        if (it != out.product_tags.end()) return static_cast<int>(it - out.product_tags.begin());
        out.product_tags.push_back(tag_of_id);
        return static_cast<int>(out.product_tags.size() - 1);
      };

      auto const& assns = *handle;
      for (std::size_t i = 0; i < assns.size(); ++i) {
        auto const& a = assns[i].first;
        auto const& b = assns[i].second;
        // left is the kind's first type: B when the product is stored swapped
        out.left.push_back(swapped ? b.key() : a.key());
        out.right.push_back(swapped ? a.key() : b.key());
        out.left_product.push_back(product_index(swapped ? b.id() : a.id()));
        out.right_product.push_back(product_index(swapped ? a.id() : b.id()));
        if constexpr (std::is_same_v<D, recob::TrackHitMeta>) {
          out.meta_index.push_back(assns.data(i).Index());
        }
        else {
          out.meta_index.push_back(-1);
        }
      }
      return true;
    }

  } // namespace

  bool readAssociation(gallery::Event const& ev, std::string const& tag, std::string const& cpp_type,
                       std::vector<std::string> const& candidate_tags, AssociationColumns& out)
  {
    // kind order (left first) as in titus/readers/products.py ASSOCIATIONS
    if (cpp_type == "art::Assns<recob::Track,recob::Hit,recob::TrackHitMeta>")
      return readAssns<recob::Track, recob::Hit, recob::TrackHitMeta>(ev, tag, false, candidate_tags, out);
    if (cpp_type == "art::Assns<recob::Hit,recob::Track,recob::TrackHitMeta>")
      return readAssns<recob::Hit, recob::Track, recob::TrackHitMeta>(ev, tag, true, candidate_tags, out);
    if (cpp_type == "art::Assns<recob::Track,recob::Hit,void>")
      return readAssns<recob::Track, recob::Hit, void>(ev, tag, false, candidate_tags, out);
    if (cpp_type == "art::Assns<recob::Hit,recob::Track,void>")
      return readAssns<recob::Hit, recob::Track, void>(ev, tag, true, candidate_tags, out);
    if (cpp_type == "art::Assns<recob::PFParticle,recob::Track,void>")
      return readAssns<recob::PFParticle, recob::Track, void>(ev, tag, false, candidate_tags, out);
    if (cpp_type == "art::Assns<recob::Track,recob::PFParticle,void>")
      return readAssns<recob::Track, recob::PFParticle, void>(ev, tag, true, candidate_tags, out);
    if (cpp_type == "art::Assns<recob::PFParticle,anab::T0,void>")
      return readAssns<recob::PFParticle, anab::T0, void>(ev, tag, false, candidate_tags, out);
    if (cpp_type == "art::Assns<anab::T0,recob::PFParticle,void>")
      return readAssns<anab::T0, recob::PFParticle, void>(ev, tag, true, candidate_tags, out);
    throw std::runtime_error("titusio: unsupported association type " + cpp_type);
  }

} // namespace titusio
