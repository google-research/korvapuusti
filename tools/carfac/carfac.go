/* partial_loudness is a package that lets you run a server presenting a web page
 * that plays sounds and lets you adjust levels.
 *
 * Run it and browse to http://localhost:12000.
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
//go:generate sh -c "c++ -c `pkg-config --libs --cflags eigen3` carfac/cpp/agc.h carfac/cpp/binaural_sai.cc carfac/cpp/car.h carfac/cpp/carfac.cc carfac/cpp/carfac_util.h carfac/cpp/common.h carfac/cpp/ear.cc carfac/cpp/ihc.h carfac/cpp/sai.cc glue/glue.cc && ar rcs libcarfac.a *.o"
package carfac

// #cgo CFLAGS: -I${SRCDIR}
// #cgo LDFLAGS: -L${SRCDIR} -lcarfac -lstdc++ -lm
// #include "carfac.h"
import "C"

import (
	"fmt"
	"reflect"
	"runtime"
	"unsafe"
)

func newFloatAry(length int) (C.float_ary, []float32) {
	floats := make([]float32, length)
	return C.float_ary{
		len:  C.int(length),
		data: (*C.float)(&floats[0]),
	}, floats
}

func floatAryToFloats(ary interface{}) []float32 {
	val := reflect.ValueOf(ary)
	var floats []float32
	header := (*reflect.SliceHeader)((unsafe.Pointer(&floats)))
	header.Cap = int(val.FieldByName("len").Int())
	header.Len = int(val.FieldByName("len").Int())
	header.Data = val.FieldByName("data").Pointer()
	return floats
}

func floatsToFloatAry(floats []float32) C.float_ary {
	return C.float_ary{
		len:  C.int(len(floats)),
		data: (*C.float)(&floats[0]),
	}
}

type CF interface {
	Run(buffer []float32)
	NAP() ([]float32, error)
	BM() ([]float32, error)
	NumChannels() int
	NumSamples() int
	SampleRate() int
	Poles() []float32
}

type carfac struct {
	numChannels int
	numSamples  int
	sampleRate  int
	poles       []float32
	openLoop    bool
	cf          *C.carfac
}

func (c *carfac) NumChannels() int {
	return c.numChannels
}

func (c *carfac) NumSamples() int {
	return c.numSamples
}

func (c *carfac) SampleRate() int {
	return c.sampleRate
}

func (c *carfac) Poles() []float32 {
	return c.poles
}

type CARFACParams struct {
	SampleRate int
	VOffset    *float64
	OpenLoop   bool
	ERBPerStep *float64
}

func New(carfacParams CARFACParams) CF {
	var vOffset *C.float
	if carfacParams.VOffset != nil {
		cVOffset := C.float(*carfacParams.VOffset)
		vOffset = &cVOffset
	}
	var erbPerStep *C.float
	if carfacParams.ERBPerStep != nil {
		cERBPerStep := C.float(*carfacParams.ERBPerStep)
		erbPerStep = &cERBPerStep
	}
	cf := C.create_carfac(C.int(carfacParams.SampleRate), vOffset, erbPerStep)
	runtime.SetFinalizer(&cf, func(i interface{}) {
		C.delete_carfac(&cf)
	})
	return &carfac{
		numChannels: int(cf.num_channels),
		numSamples:  int(cf.num_samples),
		sampleRate:  carfacParams.SampleRate,
		poles:       floatAryToFloats(cf.poles),
		openLoop:    carfacParams.OpenLoop,
		cf:          &cf,
	}
}

func (c *carfac) Run(buffer []float32) {
	var open_loop C.int
	if c.openLoop {
		open_loop = 1
	} else {
		open_loop = 0
	}
	C.carfac_run(c.cf, floatsToFloatAry(buffer), open_loop)
}

func (c *carfac) NAP() (result []float32, err error) {
	resultAry, resultFloats := newFloatAry(c.numChannels * c.numSamples)
	if errnum := C.carfac_nap(c.cf, resultAry); errnum != 0 {
		return nil, fmt.Errorf("Unable to retrieve NAP from CARFAC: %v", errnum)
	}
	return resultFloats, nil
}

func (c *carfac) BM() (result []float32, err error) {
	resultAry, resultFloats := newFloatAry(c.numChannels * c.numSamples)
	if errnum := C.carfac_bm(c.cf, resultAry); errnum != 0 {
		return nil, fmt.Errorf("Unable to retrieve BM from CARFAC: %v", errnum)
	}
	return resultFloats, nil
}
