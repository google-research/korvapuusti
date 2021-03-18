package loudness

import (
	"math"
	"testing"
)

func TestLoudInterp(t *testing.T) {
	for kHz, want := range [][]float64{
		{0.532, -31.6, 78.5}, {0.25, 0, 2.4}, {0.243, -1, -1.3}, {0.243079058405375, 2.4941907795000002, -5.777951174}, {0.242, 1.2, -5.4}, {0.242, -2.1, -1.5}, {0.24400151346964, -6.007623551, 4.293046101}, {0.24804193590836449, -9.253758445531, 9.3820408963691}, {0.254, -11.2, 12.6}, {0.2616833598672, -11.655427631899999, 13.927658963999999}, {0.271, -10.7, 13.9}, {0.2818832992122, -8.4801644679, 13.18729562}, {0.2942666363256, -5.142368415199999, 12.459818326}, {0.301, -3.1, 12.3}, {0.301, -3.1, 12.3}, {0.301, -3.1, 12.3}, {0.301, -3.1, 12.3}, {0.301, -3.1, 12.3}, {0.301, -3.1, 12.3}, {0.301, -3.1, 12.3},
	} {
		if a, b, c := loudInterp(float64(kHz) * 1000); a != want[0] || b != want[1] || c != want[2] {
			t.Errorf("loudInterp(%v) produced %v, %v, %v - wanted %+v", float64(kHz)*1000, a, b, c, want)
		}
	}
}

func TestConversion(t *testing.T) {
	for _, tst := range []struct {
		f     float64
		spl   float64
		phons float64
	}{
		{
			3990.5247,
			0,
			6.8323366176854705,
		},
		{
			1000,
			10,
			10,
		},
		{
			1000,
			30,
			30,
		},
		{
			1000,
			70,
			70,
		},
		{
			63,
			37,
			2.196123316489974,
		},
		{
			63,
			60,
			21.811179536451718,
		},
		{
			63,
			80,
			50.65777743557178,
		},
		{
			4000,
			20,
			24.45505550307739,
		},
		{
			4000,
			50,
			52.76335647102224,
		},
		{
			4000,
			80,
			81.66233918735878,
		},
	} {
		if got := SPL2Phons(tst.spl, tst.f); math.Round(10*got) != math.Round(10*tst.phons) {
			t.Errorf("SPL2Phons(%v, %v) produced %v, wanted %v", tst.f, tst.spl, got, tst.phons)
		}
		if got := Phons2SPL(tst.phons, tst.f); math.Round(10*got) != math.Round(10*tst.spl) {
			t.Errorf("Phons2SPL(%v, %v) produced %v, wanted %v", tst.f, tst.phons, got, tst.spl)
		}
	}
}
