package codec

import (
	"strings"
	"testing"
)

// The routes are a table of measured control ids, and a typo in one of them is
// silence rather than an error — the same reason the jack routing table is
// pinned. Both ends are covered because they failed independently: the capture
// routes leave the ADCs powered down, the playback routes leave the DAC powered
// down, and either alone is a device that looks healthy in every log it writes.
//
// The ids here are the FireOS 5 board's. They are the fallback and the
// documentation; what a device actually writes is resolved by NAME at runtime.
func TestRoutesCoverBothEndsOfTheAudioPath(t *testing.T) {
	want := map[string]string{
		// capture: DIF1 into all four ADCs, left and right
		"170": "ADC_D Right Ip Select ADC_D DIF1_R switch",
		"177": "ADC_D Left Ip Select ADC_D DIF1_L switch",
		"184": "ADC_C Right Ip Select ADC_C DIF1_R switch",
		"191": "ADC_C Left Ip Select ADC_C DIF1_L switch",
		"200": "ADC_B Right Ip Select ADC_B DIF1_R switch",
		"207": "ADC_B Left Ip Select ADC_B DIF1_L switch",
		"216": "ADC_A Right Ip Select ADC_A DIF1_R switch",
		"223": "ADC_A Left Ip Select ADC_A DIF1_L switch",
		// playback: DAC into the output mixer
		"234": "HPR Output Mixer R_DAC Switch",
		"237": "HPL Output Mixer L_DAC Switch",
	}

	got := map[string]string{}
	for _, w := range Routes {
		if _, dup := got[w.Ctl]; dup {
			t.Errorf("control %s listed twice", w.Ctl)
		}
		if w.Value != "1" {
			t.Errorf("control %s (%s): value %q, want \"1\" — every route here is a switch to close",
				w.Ctl, w.Name, w.Value)
		}
		got[w.Ctl] = w.Name
	}

	for ctl, name := range want {
		if got[ctl] != name {
			t.Errorf("control %s: name %q, want %q", ctl, got[ctl], name)
		}
	}
	for ctl := range got {
		if _, ok := want[ctl]; !ok {
			t.Errorf("unexpected control %s (%s)", ctl, got[ctl])
		}
	}
}

// Real `tinymix -D 0` output, trimmed to the rows that matter, captured
// 2026-09-16 from two Dots running emOS side by side on one desk.
//
// The padding is verbatim and must stay that way: the longest names here leave
// a SINGLE space before the value, indistinguishable from a space inside a
// name. That is why resolveIDs cuts the name at the column the header declares
// rather than looking for a gap. Reflowing this fixture would hide the case it
// exists for.
const (
	fireos5Dump = "Mixer name: 'mt-snd-card'\n" +
		"Number of controls: 239\n" +
		"ctl\ttype\tnum\tname                                     value\n" +
		"170\tBOOL\t1\tADC_D Right Ip Select ADC_D DIF1_R switch On\n" +
		"223\tBOOL\t1\tADC_A Left Ip Select ADC_A DIF1_L switch On\n" +
		"232\tBOOL\t1\tLeft Input Mixer IN3_L P Switch          Off\n" +
		"233\tBOOL\t1\tLOR Output Mixer R_DAC Switch            Off\n" +
		"234\tBOOL\t1\tHPR Output Mixer R_DAC Switch            On\n" +
		"235\tBOOL\t1\tHPR Output Mixer IN1_R Switch            Off\n" +
		"236\tBOOL\t1\tLOL Output Mixer L_DAC Switch            Off\n" +
		"237\tBOOL\t1\tHPL Output Mixer L_DAC Switch            On\n"

	fireos6Dump = "Mixer name: 'mt-snd-card'\n" +
		"Number of controls: 241\n" +
		"ctl\ttype\tnum\tname                                     value\n" +
		"170\tBOOL\t1\tADC_D Right Ip Select ADC_D IN2_R switch On\n" +
		"172\tBOOL\t1\tADC_D Right Ip Select ADC_D DIF1_R switch Off\n" +
		"223\tBOOL\t1\tADC_A Left Ip Select ADC_A IN2_L switch On\n" +
		"225\tBOOL\t1\tADC_A Left Ip Select ADC_A DIF1_L switch Off\n" +
		"234\tBOOL\t1\tLeft Input Mixer IN3_L P Switch          On\n" +
		"235\tBOOL\t1\tLOR Output Mixer R_DAC Switch            Off\n" +
		"236\tBOOL\t1\tHPR Output Mixer R_DAC Switch            Off\n" +
		"237\tBOOL\t1\tHPR Output Mixer IN1_R Switch            On\n" +
		"238\tBOOL\t1\tLOL Output Mixer L_DAC Switch            Off\n" +
		"239\tBOOL\t1\tHPL Output Mixer L_DAC Switch            Off\n"
)

// synth builds a listing with the same column layout tinymix emits, so the
// synthetic cases below cannot drift from the real fixtures by a space.
// nameWidth is the real one: the longest control on these boards is 41
// characters and leaves a single space before its value.
const nameWidth = 41

type row struct{ id, name, value string }

func synth(rows ...row) string {
	b := "Mixer name: 'mt-snd-card'\nctl\ttype\tnum\t" +
		"name" + strings.Repeat(" ", nameWidth-len("name")) + "value\n"
	for _, r := range rows {
		pad := nameWidth - len(r.name)
		if pad < 1 {
			pad = 1
		}
		b += r.id + "\tBOOL\t1\t" + r.name + strings.Repeat(" ", pad) + r.value + "\n"
	}
	return b
}

func names() []string {
	out := make([]string, 0, len(Routes))
	for _, w := range Routes {
		out = append(out, w.Name)
	}
	return out
}

// The FireOS 5 board is the one the table was measured on, so resolving by
// name has to reproduce the hardcoded ids exactly. Anything else would mean
// the change moves the fielded fleet, which is the outcome that matters most
// here — every device in the field today is FireOS 5.
func TestResolvingByNameMatchesTheMeasuredIDsOnFireOS5(t *testing.T) {
	got := resolveIDs(fireos5Dump, names())
	for _, w := range Routes {
		id, ok := got[w.Name]
		if !ok {
			continue // not in this trimmed fixture
		}
		if id != w.Ctl {
			t.Errorf("%q resolved to ctl %s on FireOS 5, want the measured %s",
				w.Name, id, w.Ctl)
		}
	}
	if got["HPR Output Mixer R_DAC Switch"] != "234" {
		t.Errorf("playback route resolved to %q, want 234",
			got["HPR Output Mixer R_DAC Switch"])
	}
}

// #546. The same names are two places later on FireOS 6, and the ids the table
// carries belong to other controls there — writing to them is a valid write
// that silently does the wrong thing.
func TestResolvingByNameFindsTheShiftedIDsOnFireOS6(t *testing.T) {
	got := resolveIDs(fireos6Dump, names())

	for name, want := range map[string]string{
		"HPR Output Mixer R_DAC Switch":             "236",
		"HPL Output Mixer L_DAC Switch":             "239",
		"ADC_D Right Ip Select ADC_D DIF1_R switch": "172",
		"ADC_A Left Ip Select ADC_A DIF1_L switch":  "225",
	} {
		if got[name] != want {
			t.Errorf("%q resolved to %q on FireOS 6, want %s", name, got[name], want)
		}
	}

	// The failure this whole change exists to stop: the hardcoded playback id
	// is a real control on this board, just not the one we mean.
	if got["HPR Output Mixer R_DAC Switch"] == "234" {
		t.Error("resolved to 234 on FireOS 6, which is Left Input Mixer IN3_L " +
			"P Switch there — the silent-audio bug")
	}
}

// A name that merely STARTS another control's name must not match it. Both
// boards carry "HPR Output Mixer R_DAC Switch" alongside "HPR Output Mixer
// IN1_R Switch", and the capture names differ only in their input token.
func TestAPrefixOfALongerNameDoesNotMatchIt(t *testing.T) {
	dump := synth(
		row{"10", "Output Mixer Switch Extra", "Off"},
		row{"11", "Output Mixer Switch", "On"},
	)
	got := resolveIDs(dump, []string{"Output Mixer Switch"})
	if got["Output Mixer Switch"] != "11" {
		t.Errorf("resolved to %q, want 11 — a longer control that starts the "+
			"same way must not satisfy the lookup", got["Output Mixer Switch"])
	}
}

// Two controls answering to one name is something to report rather than pick
// between, so the lookup yields nothing and EnsureRoutes logs and skips.
func TestAnAmbiguousNameResolvesToNothing(t *testing.T) {
	dump := synth(
		row{"10", "Duplicated Switch", "Off"},
		row{"11", "Duplicated Switch", "Off"},
	)
	if got := resolveIDs(dump, []string{"Duplicated Switch"}); len(got) != 0 {
		t.Errorf("ambiguous name resolved to %v, want no answer", got)
	}
}

// An unreadable or empty listing must yield nothing, so EnsureRoutes takes the
// fallback rather than acting on a half-read mixer.
func TestAnEmptyListingResolvesToNothing(t *testing.T) {
	if got := resolveIDs("", names()); len(got) != 0 {
		t.Errorf("empty dump resolved to %v, want no answer", got)
	}
}
