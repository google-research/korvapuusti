/* leaktest stress tests the carfac package to make it simpler to see if
 * there's a memory leak while running it.
 *
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 */
package main

import (
	"math"
	"runtime"
	"sync"

	"github.com/cheggaaa/pb"
	"github.com/google-research/korvapuusti/tools/carfac"
)

func makeSignal(l int) []float32 {
	buf := make([]float32, l)
	f := 1000.0
	ti := 0.0
	step := 1.0 / 48000.0
	for idx := range buf {
		buf[idx] = float32(math.Sin(2 * math.Pi * f * ti * step))
		ti += step
	}
	return buf
}

func main() {
	iters := 1000000
	perCPU := iters / runtime.NumCPU()
	bar := pb.StartNew(perCPU * runtime.NumCPU()).Prefix("Running")
	wg := &sync.WaitGroup{}
	for p := 0; p < runtime.NumCPU(); p++ {
		wg.Add(1)
		go func() {
			for i := 0; i < perCPU; i++ {
				cf := carfac.New(carfac.CARFACParams{SampleRate: 48000})
				sig := makeSignal(cf.NumSamples())
				cf.Run(sig)
				result, err := cf.BM()
				if err != nil {
					panic(err)
				}
				if len(result) == 0 {
					panic("huh")
				}
				bar.Increment()
			}
			wg.Done()
		}()
	}
	wg.Wait()
	bar.Finish()
}
