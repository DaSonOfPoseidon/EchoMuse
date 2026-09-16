// Package codec brings up the DAPM routes the audio path needs, instead of
// inheriting them from Amazon's audio HAL.
//
// Until 2026-09-04 nothing here existed, because nothing had to: on FireOS the
// HAL configures the codec long before our process opens a PCM, so both the
// microphone array and the speaker worked and we never learned we were relying
// on it. Running EchoMuse on a device with no Android userspace made the
// dependency visible in the least helpful way available — capture returned a
// steady rms≈0.00035 with a perfectly healthy ALSA clock (300.6s of audio over
// 300.3s of wall, zero stalls), and playback reported
// "voice stream complete, periods=35 underruns=0" while the room stayed silent.
//
// Both halves had the same cause. An ASoC route that is not connected leaves
// DAPM no reason to power the converter at either end of it, so the hardware is
// powered DOWN rather than merely misrouted. Read off the codec (i2c 2-0018) on
// a device with no Android, against a stock FireOS Dot running the same
// firmware:
//
//	          stock   ours     meaning
//	  0012      85      05     NADC clock divider, bit7 = powered
//	  0013      83      03     MADC clock divider
//	  0026      11      00     ADC flags: left+right converting
//	  003f      d6      16     DAC data path, bit7/6 = left/right powered
//	  0089      30      00     output driver power
//	  008c/8d   08      00     HPL/HPR output mixer routing
//
// Applying the routes below took every one of those to stock's exact value.
//
// This is written unconditionally, on FireOS as well, and that is deliberate:
// the values are the ones the HAL would set anyway, so a device that still has
// Android loses nothing, and one that does not gains a working audio path. The
// point of the project is not to need Amazon's userspace, and inheriting
// hardware state from it is a dependency whether or not it currently holds.
package codec

import (
	"log"
	"os/exec"
	"strings"
	"sync"
)

// Write is one `tinymix -D 0 <ctl> <value>` invocation.
//
// NAME is authoritative and Ctl is the id measured on a FireOS 5 board, kept
// as documentation and as the fallback for a mixer whose control list cannot
// be read at all.
type Write struct {
	Ctl   string
	Value string
	Name  string // the control's mixer name — what this route actually means
}

// Routes is every DAPM switch that must be closed for audio to flow.
//
// RESOLVED BY NAME AT RUNTIME. The ids below are what a FireOS 5 board gives
// these controls; a FireOS 6 board gives them different ones, because its
// kernel exposes two extra controls early in the list and everything after
// shifts by two. Measured 2026-09-16 on two Dots running emOS side by side:
//
//	                                 FireOS 5   FireOS 6
//	  controls in the mixer             239        241
//	  HPR Output Mixer R_DAC Switch     234        236
//	  ADC_A Left ... DIF1_L switch      223        225
//
// So on every FireOS 6 device all ten writes landed two places early. 234 set
// "Left Input Mixer IN3_L P Switch" and the DAC was never connected to the
// output mixer, which is silence; the eight capture writes set the
// single-ended IN2 inputs while the microphone array is on the differential
// DIF1 ones. Reported as #546 by @jthoward64 and reproduced here.
//
// IT FAILS SILENTLY BY CONSTRUCTION, which is why it took a user to find it:
// writing 1 to the WRONG control is a perfectly valid write, so tinymix exits
// 0, the failure count stays 0, and the "audio may be silent" warning below
// never fires. The device logs a clean boot and plays nothing. Resolving by
// name is what makes a mismatch loud — a name that is not in the mixer's own
// list is reported, where a wrong number never could be.
//
// The whole fleet was FireOS 5 until amonet v2 made FireOS 6 devices usable,
// which is why this shipped working and broke for new users only.
//
// CAPTURE: the microphone array reaches the codec on the DIFFERENTIAL inputs,
// not the single-ended ones. Nothing routed DIF1 into any of the four ADCs, so
// all four sat powered down. Note the neighbouring "ADC_x DIF1_L/R Input Gain"
// controls are a DIFFERENT thing in the same register block and were the first
// thing tried; changing them does nothing, and they are not listed here.
//
// PLAYBACK: the DAC was not connected to the output mixer, so it powered down
// with the firmware streaming correctly into it.
var Routes = []Write{
	{"170", "1", "ADC_D Right Ip Select ADC_D DIF1_R switch"},
	{"177", "1", "ADC_D Left Ip Select ADC_D DIF1_L switch"},
	{"184", "1", "ADC_C Right Ip Select ADC_C DIF1_R switch"},
	{"191", "1", "ADC_C Left Ip Select ADC_C DIF1_L switch"},
	{"200", "1", "ADC_B Right Ip Select ADC_B DIF1_R switch"},
	{"207", "1", "ADC_B Left Ip Select ADC_B DIF1_L switch"},
	{"216", "1", "ADC_A Right Ip Select ADC_A DIF1_R switch"},
	{"223", "1", "ADC_A Left Ip Select ADC_A DIF1_L switch"},

	{"234", "1", "HPR Output Mixer R_DAC Switch"},
	{"237", "1", "HPL Output Mixer L_DAC Switch"},
}

var once sync.Once

// resolveIDs maps each wanted control NAME to the id this board gives it, from
// the mixer's own listing. Pure, so the board difference is testable off-target
// against captured output from both kernels.
//
// A line is "<id>\t<type>\t<num>\t<name><padding><value>", and where the name
// ends is NOT recoverable from the padding: the longest names leave a single
// space before the value, exactly like the space inside a name. Prefix
// matching therefore cannot work — "HPR Output Mixer R_DAC Switch" is a prefix
// of no other control here today, but nothing stops the next kernel adding one,
// and the failure would be another silent wrong write.
//
// The header line declares the column: "ctl\ttype\tnum\tname<pad>value". The
// offset of "value" within its fourth field is the width the names are padded
// to, so the name can be cut exactly and compared whole. No header means no
// answer, and EnsureRoutes falls back to the measured ids.
//
// A name matching more than one line is dropped rather than guessed at: two
// controls answering to one name is something to report, not to pick between.
func resolveIDs(dump string, names []string) map[string]string {
	want := make(map[string]bool, len(names))
	for _, n := range names {
		want[n] = true
	}

	width := -1
	hits := make(map[string][]string, len(names))
	for _, line := range strings.Split(dump, "\n") {
		f := strings.SplitN(strings.TrimRight(line, "\r"), "\t", 4)
		if len(f) < 4 {
			continue
		}
		if width < 0 {
			// The header names its own columns; anything before it is the
			// mixer name and control count.
			if strings.TrimSpace(f[0]) == "ctl" {
				if i := strings.Index(f[3], "value"); i > 0 {
					width = i
				}
			}
			continue
		}
		if len(f[3]) < width {
			continue
		}
		name := strings.TrimRight(f[3][:width], " \t")
		if want[name] {
			hits[name] = append(hits[name], strings.TrimSpace(f[0]))
		}
	}

	out := make(map[string]string, len(names))
	for n, ids := range hits {
		if len(ids) == 1 {
			out[n] = ids[0]
		}
	}
	return out
}

// EnsureRoutes applies Routes exactly once per process.
//
// Called from both the microphone and the speaker Init, because either may run
// first and each needs the routes closed BEFORE it opens its PCM — DAPM decides
// what to power at stream open. The sync.Once is what makes calling it from
// both sites free: process spawns are not cheap on this hardware (a heavy shell
// command was observed inducing mic capture stalls, and the mic pipeline has a
// hard 160ms deadline), so ten of them must not become twenty. The listing adds
// exactly one more spawn, once.
func EnsureRoutes() {
	once.Do(func() {
		names := make([]string, 0, len(Routes))
		for _, w := range Routes {
			names = append(names, w.Name)
		}

		// FAILURE TO LOOK IS NOT EVIDENCE. A mixer we cannot list tells us
		// nothing about this board, so fall back to the measured FireOS 5 ids
		// — the behaviour every fielded device has today — rather than
		// refusing to route anything and guaranteeing silence.
		var byName map[string]string
		dump, err := exec.Command("tinymix", "-D", "0").Output()
		if err != nil {
			log.Printf("[codec] could not list mixer controls (%v) — falling back "+
				"to the measured FireOS 5 control ids", err)
		} else {
			byName = resolveIDs(string(dump), names)
		}

		var failed int
		for _, w := range Routes {
			ctl := w.Ctl
			if byName != nil {
				id, ok := byName[w.Name]
				if !ok {
					// The listing was read and this control is not in it.
					// That IS evidence, unlike the case above, so say so and
					// skip rather than write to a number this board gives to
					// something else.
					failed++
					log.Printf("[codec] route %q is not in this mixer's control "+
						"list — skipping (id %s belongs to another control here)",
						w.Name, w.Ctl)
					continue
				}
				if id != w.Ctl {
					log.Printf("[codec] route %q is ctl %s on this board, not %s",
						w.Name, id, w.Ctl)
				}
				ctl = id
			}
			out, err := exec.Command("tinymix", "-D", "0", ctl, w.Value).CombinedOutput()
			if err != nil {
				failed++
				log.Printf("[codec] route ctl %s (%s): %v — %s",
					ctl, w.Name, err, strings.TrimSpace(string(out)))
			}
		}
		if failed > 0 {
			log.Printf("[codec] %d of %d DAPM routes failed — audio may be silent",
				failed, len(Routes))
		}
	})
}
