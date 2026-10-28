#include "ProductReader.h"

#include <algorithm>
#include <stdexcept>

#include "canvas/Utilities/InputTag.h"
#include "gallery/Event.h"
#include "lardataobj/RawData/RawDigit.h"
#include "lardataobj/RawData/raw.h"
#include "lardataobj/RecoBase/Hit.h"
#include "lardataobj/RecoBase/Wire.h"

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
    auto const handle = ev.getHandle<std::vector<recob::Hit>>(art::InputTag(tag));
    if (!handle.isValid()) return false;

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
      auto const handle = ev.getHandle<std::vector<T>>(art::InputTag(tag));
      if (!handle.isValid()) return false;

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
    auto const handle = ev.getHandle<std::vector<raw::RawDigit>>(art::InputTag(tag));
    if (!handle.isValid()) return false;

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

} // namespace titusio
