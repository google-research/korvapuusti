/* workerpool contains code to run a limited number of error handling goroutines concurrently.
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
package workerpool

import (
	"fmt"
	"sync"
	"sync/atomic"
)

// MultiErr contains multiple errors.
type MultiErr []error

// Error returns a string representation of the multi error.
func (m MultiErr) Error() string {
	return fmt.Sprint([]error(m))
}

// WorkerPool runs a limited number of error handling goroutines concurrently.
type WorkerPool struct {
	queues            map[string]chan func() error
	errors            map[string]chan error
	runningGoRoutines int64
	queuedJobs        int64
	runningJobs       int64
	finishedJobs      int64
}

// Queue will run the function in the provided queue.
func (w *WorkerPool) Queue(queue string, f func() error) {
	atomic.AddInt64(&w.queuedJobs, 1)
	w.queues[queue] <- f
}

// QueuedJobs returns the number of queued jobs since creation.
func (w *WorkerPool) QueuedJobs() int64 {
	return atomic.LoadInt64(&w.queuedJobs)
}

// RunningJobs returns the current number of running jobs.
func (w *WorkerPool) RunningJobs() int64 {
	return atomic.LoadInt64(&w.runningJobs)
}

// FinishedJobs returns the number of finished jobs since creation.
func (w *WorkerPool) FinishedJobs() int64 {
	return atomic.LoadInt64(&w.finishedJobs)
}

// RunningGoRoutines returns the number of currently running go routines
func (w *WorkerPool) RunningGoRoutines() int64 {
	atomic.AddInt64(&w.queuedJobs, 1)
	return atomic.LoadInt64(&w.runningGoRoutines)
}

// Close stops accepting new jobs for queue. Queue for the same queue will panic after this is called.
func (w *WorkerPool) Close(queue string) {
	close(w.queues[queue])
}

// Wait waits for the queues to be closed and all submitted jobs to finish and returns the errors.
func (w *WorkerPool) Wait(queues ...string) error {
	if len(queues) == 0 {
		return fmt.Errorf("Wait called with no queues, did you mean WaitAll?")
	}
	me := MultiErr{}
	for _, queue := range queues {
		for err := range w.errors[queue] {
			if err != nil {
				me = append(me, err)
			}
		}
	}
	if len(me) == 0 {
		return nil
	}
	return me
}

// WaitAll waits for all queues to be closed and all submitted jobs to finish and returns the errors.
func (w *WorkerPool) WaitAll() error {
	me := MultiErr{}
	for _, errors := range w.errors {
		for err := range errors {
			if err != nil {
				me = append(me, err)
			}
		}
	}
	if len(me) == 0 {
		return nil
	}
	return me
}

func (w *WorkerPool) run(f func()) {
	atomic.AddInt64(&w.runningGoRoutines, 1)
	go func() {
		defer atomic.AddInt64(&w.runningGoRoutines, -1)
		f()
	}()
}

func (w *WorkerPool) runWait(wg *sync.WaitGroup, f func()) {
	wg.Add(1)
	w.run(func() {
		defer wg.Done()
		f()
	})
}

// New returns a new worker pool.
//
// queues contains a map of queue names to queue concurrency limits, where 0 means no limit.
//
// Close needs to be called for each queue or there will be hanging goroutines.
func New(queues map[string]int) *WorkerPool {
	w := &WorkerPool{
		queues: map[string]chan func() error{},
		errors: map[string]chan error{},
	}
	for queue := range queues {
		w.queues[queue] = make(chan func() error)
		w.errors[queue] = make(chan error)
	}

	for iterQueue, iterLimit := range queues {
		queue := iterQueue
		limit := iterLimit
		w.run(func() {
			defer close(w.errors[queue])
			wg := &sync.WaitGroup{}
			w.runWait(wg, func() {
				tickets := make(chan struct{}, limit)
				for iterJob := range w.queues[queue] {
					job := iterJob
					if limit > 0 {
						tickets <- struct{}{}
					}
					w.runWait(wg, func() {
						if limit > 0 {
							defer func() {
								<-tickets
							}()
						}
						defer atomic.AddInt64(&w.finishedJobs, 1)
						atomic.AddInt64(&w.runningJobs, 1)
						defer atomic.AddInt64(&w.runningJobs, -1)
						err := job()
						w.runWait(wg, func() {
							w.errors[queue] <- err
						})
					})
				}
			})
			wg.Wait()
		})
	}
	return w
}
