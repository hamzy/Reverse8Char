/*
 * This work is marked with CC0 1.0 Universal. 
 * To view a copy of this license, visit:
 * http://creativecommons.org
 *
 * No Copyright: The person who associated a work with this deed has dedicated 
 * the work to the public domain by waiving all of his or her rights to the work 
 * worldwide under copyright law, including all related and neighboring rights, 
 * to the extent allowed by law.
 */

// https://share.google/aimode/7VvT09F5IOnQtqu7Z

// $ go mod init example.com/Reverse8Char/v2
// $ go mod tidy
// $ go build

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

const (
	Alphabet  = "abcdefghijklmnopqrstuvwxyz"
	AlphaLen  = len(Alphabet)
	StrLen    = 8
	InputDim  = StrLen * AlphaLen
	BatchSize = 32
)

type Neuron struct {
	Weights []float64
	Bias    float64
}

type MLP struct {
	Hidden []*Neuron
	Output []*Neuron
	mu     sync.Mutex
}

type Job struct {
	Input  []float64
	Target []float64
	LR     float64
}

// --- MATH UTILITIES ---

func sigmoid(x float64) float64      { return 1.0 / (1.0 + math.Exp(-math.Max(-20, math.Min(20, x)))) }
func sigmoidDeriv(s float64) float64 { return s * (1.0 - s) }

func Softmax(logits []float64) []float64 {
	max := -math.MaxFloat64
	for _, v := range logits {
		if v > max { max = v }
	}
	res, sum := make([]float64, len(logits)), 0.0
	for i, v := range logits {
		res[i] = math.Exp(v - max)
		sum += res[i]
	}
	for i := range res { res[i] /= (sum + 1e-15) }
	return res
}

func NewMLP(in, hid, out int) *MLP {
	m := &MLP{Hidden: make([]*Neuron, hid), Output: make([]*Neuron, out)}
	limitH := math.Sqrt(6.0 / float64(in+hid))
	for i := range m.Hidden {
		w := make([]float64, in)
		for j := range w { w[j] = (rand.Float64()*2 - 1) * limitH }
		m.Hidden[i] = &Neuron{Weights: w}
	}
	limitO := math.Sqrt(6.0 / float64(hid+out))
	for i := range m.Output {
		w := make([]float64, hid)
		for j := range w { w[j] = (rand.Float64()*2 - 1) * limitO }
		m.Output[i] = &Neuron{Weights: w}
	}
	return m
}

// --- CORE LOGIC ---

func (m *MLP) Predict(s string) string {
	in := encode(s)
	h := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden {
		var sum float64
		for j, v := range in { sum += v * n.Weights[j] }
		h[i] = sigmoid(sum + n.Bias)
	}
	res := ""
	for i := 0; i < StrLen; i++ {
		logits := make([]float64, AlphaLen)
		for j := 0; j < AlphaLen; j++ {
			var sum float64
			for k, v := range h { sum += v * m.Output[i*AlphaLen+j].Weights[k] }
			logits[j] = sum + m.Output[i*AlphaLen+j].Bias
		}
		p := Softmax(logits)
		best := 0
		for idx, val := range p {
			if val > p[best] { best = idx }
		}
		res += string(Alphabet[best])
	}
	return res
}

func (m *MLP) TrainBatch(batch []Job) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, job := range batch {
		// 1. Forward
		hActs := make([]float64, len(m.Hidden))
		for i, n := range m.Hidden {
			var sum float64
			for j, v := range job.Input { sum += v * n.Weights[j] }
			hActs[i] = sigmoid(sum + n.Bias)
		}
		oSums := make([]float64, len(m.Output))
		for i, n := range m.Output {
			var sum float64
			for j, v := range hActs { sum += v * n.Weights[j] }
			oSums[i] = sum + n.Bias
		}

		// 2. Gradients
		oGrads := make([]float64, len(m.Output))
		for i := 0; i < StrLen; i++ {
			probs := Softmax(oSums[i*AlphaLen : (i+1)*AlphaLen])
			for j := 0; j < AlphaLen; j++ {
				idx := i*AlphaLen + j
				oGrads[idx] = probs[j] - job.Target[idx]
			}
		}

		hGrads := make([]float64, len(m.Hidden))
		for i := range m.Hidden {
			var err float64
			for j := range m.Output {
				err += oGrads[j] * m.Output[j].Weights[i]
			}
			hGrads[i] = err * sigmoidDeriv(hActs[i])
		}

		// 3. Update
		lr := job.LR / float64(len(batch))
		for i, n := range m.Output {
			for j := range n.Weights { n.Weights[j] -= hActs[j] * oGrads[i] * lr }
			n.Bias -= oGrads[i] * lr
		}
		for i, n := range m.Hidden {
			for j := range n.Weights { n.Weights[j] -= job.Input[j] * hGrads[i] * lr }
			n.Bias -= hGrads[i] * lr
		}
	}
}

// --- DATA HELPERS ---

func encode(s string) []float64 {
	v := make([]float64, InputDim)
	for i := 0; i < len(s) && i < StrLen; i++ {
		for j, char := range Alphabet {
			if byte(char) == s[i] {
				v[i*AlphaLen+j] = 1.0
				break
			}
		}
	}
	return v
}

func reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

func generateRandomString(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = Alphabet[rand.Intn(AlphaLen)]
	}
	return string(b)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	model := NewMLP(InputDim, 128, InputDim)
	jobs := make(chan Job, 100)
	var wg sync.WaitGroup

	// Workers
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var batch []Job
			for j := range jobs {
				batch = append(batch, j)
				if len(batch) >= BatchSize {
					model.TrainBatch(batch)
					batch = nil
				}
			}
		}()
	}

	fmt.Println("Training...")
	start := time.Now()
	for e := 1; e <= 200000; e++ {
		s := generateRandomString(8)
		lr := 0.2 * math.Pow(0.9, float64(e/25000))
		jobs <- Job{encode(s), encode(reverse(s)), lr}

		if e%10000 == 0 {
			test := "onionsys"
			pred := model.Predict(test)
			fmt.Printf("Step %d | %s -> %s\n", e, test, pred)
			if pred == "sysnoino" {
				fmt.Println("Success! Stopping early.")
				break
			}
		}
	}
	close(jobs)
	wg.Wait()
	fmt.Printf("\nFinal Check: onionsys -> %s\nTime: %v\n", model.Predict("onionsys"), time.Since(start))
}
