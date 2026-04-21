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
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"
)

const (
	Alphabet = "abcdefghijklmnopqrstuvwxyz"
	AlphaLen = len(Alphabet)
	StrLen   = 8
	InputDim = StrLen * AlphaLen // 208
)

type Neuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

type MLP struct {
	Hidden []*Neuron  `json:"hidden"`
	Output []*Neuron  `json:"output"`
	mu     sync.Mutex
}

type Job struct {
	Input  []float64
	Target []float64
	LR     float64
}

type Candidate struct {
	Char byte
	Prob float64
}

// --- MATH UTILITIES ---

func sigmoid(x float64) float64           { return 1.0 / (1.0 + math.Exp(-x)) }
func sigmoidDerivative(s float64) float64  { return s * (1.0 - s) }

func (n *Neuron) Process(inputs []float64) float64 {
	var sum float64
	for i, v := range inputs { sum += v * n.Weights[i] }
	return sigmoid(sum + n.Bias)
}

func Softmax(logits []float64) []float64 {
	var max float64 = -math.MaxFloat64
	for _, v := range logits { if v > max { max = v } }
	res := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		res[i] = math.Exp(v - max)
		sum += res[i]
	}
	for i := range res { res[i] /= sum }
	return res
}

// --- CORE LOGIC ---

func NewMLP(inputDim, hiddenDim, outputDim int) *MLP {
	m := &MLP{
		Hidden: make([]*Neuron, hiddenDim),
		Output: make([]*Neuron, outputDim),
	}
	for i := range m.Hidden {
		m.Hidden[i] = &Neuron{Weights: randomWeights(inputDim), Bias: rand.NormFloat64() * 0.1}
	}
	for i := range m.Output {
		m.Output[i] = &Neuron{Weights: randomWeights(hiddenDim), Bias: rand.NormFloat64() * 0.1}
	}
	return m
}

func randomWeights(n int) []float64 {
	w := make([]float64, n)
	for i := range w { w[i] = rand.NormFloat64() * 0.1 }
	return w
}

func (m *MLP) TrainCE(inputs []float64, targets []float64, lr float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	const lambda = 0.0001 // L2 Regularization strength

	// 1. Forward
	hOutputs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { hOutputs[i] = n.Process(inputs) }
	rawOutputs := make([]float64, len(m.Output))
	for i, n := range m.Output { rawOutputs[i] = n.Process(hOutputs) }

	// 2. Gradients (Softmax + Cross-Entropy)
	oGradients := make([]float64, len(m.Output))
	for i := 0; i < StrLen; i++ {
		start, end := i*AlphaLen, (i+1)*AlphaLen
		probs := Softmax(rawOutputs[start:end])
		for j := 0; j < AlphaLen; j++ {
			idx := start + j
			oGradients[idx] = targets[idx] - probs[j]
		}
	}

	hGradients := make([]float64, len(m.Hidden))
	for i := range m.Hidden {
		var err float64
		for j := range m.Output { err += oGradients[j] * m.Output[j].Weights[i] }
		hGradients[i] = err * sigmoidDerivative(hOutputs[i])
	}

	// 3. Update with L2 Weight Decay
	for i, n := range m.Output {
		for j := range n.Weights {
			n.Weights[j] = n.Weights[j]*(1-lr*lambda) + (hOutputs[j] * oGradients[i] * lr)
		}
		n.Bias += oGradients[i] * lr
	}
	for i, n := range m.Hidden {
		for j := range n.Weights {
			n.Weights[j] = n.Weights[j]*(1-lr*lambda) + (inputs[j] * hGradients[i] * lr)
		}
		n.Bias += hGradients[i] * lr
	}
}

func (m *MLP) Validate(count int) float64 {
	correct := 0
	for i := 0; i < count; i++ {
		test := generateRandomString(StrLen)
		target := reverse(test)
		prediction := m.PredictRaw(encode(test))

		match := true
		for pos := 0; pos < StrLen; pos++ {
			block := prediction[pos*AlphaLen : (pos+1)*AlphaLen]
			best := 0
			for j := range block { if block[j] > block[best] { best = j } }
			if Alphabet[best] != target[pos] {
				match = false
				break
			}
		}
		if match { correct++ }
	}
	return (float64(correct) / float64(count)) * 100
}

func (m *MLP) PredictRaw(inputs []float64) []float64 {
	h := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { h[i] = n.Process(inputs) }
	res := make([]float64, len(m.Output))
	for i, n := range m.Output { res[i] = n.Process(h) }
	return res
}

// --- DATA HELPERS ---

func encode(s string) []float64 {
	v := make([]float64, InputDim)
	for i := 0; i < StrLen; i++ {
		for j := range Alphabet {
			if Alphabet[j] == s[i] { v[i*AlphaLen+j] = 1.0; break }
		}
	}
	return v
}

func reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 { r[i], r[j] = r[j], r[i] }
	return string(r)
}

func generateRandomString(n int) string {
	b := make([]byte, n)
	for i := range b { b[i] = Alphabet[rand.Intn(AlphaLen)] }
	return string(b)
}

func displayResults(vec []float64) {
	for i := 0; i < StrLen; i++ {
		block := Softmax(vec[i*AlphaLen : (i+1)*AlphaLen])
		cands := make([]Candidate, AlphaLen)
		for j := range Alphabet { cands[j] = Candidate{Alphabet[j], block[j]} }
		sort.Slice(cands, func(a, b int) bool { return cands[a].Prob > cands[b].Prob })
		fmt.Printf("Pos %d: [%c: %.1f%%]  ", i, cands[0].Char, cands[0].Prob*100)
	}
	fmt.Println()
}

func main() {
	rand.Seed(time.Now().UnixNano())
	const file = "final_weights.json"
	var model *MLP

	if _, err := os.Stat(file); err == nil {
		fmt.Println("Loading trained model...")
		d, _ := os.ReadFile(file)
		json.Unmarshal(d, &model)
	} else {
		model = NewMLP(InputDim, 128, InputDim)
		const workers, total = 8, 200000
		jobs := make(chan Job, 100)
		var wg sync.WaitGroup

		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := range jobs { model.TrainCE(j.Input, j.Target, j.LR) }
			}()
		}

		fmt.Println("Training with all features enabled...")
		for i := 1; i <= total; i++ {
			s := generateRandomString(8)
			lr := 0.2 * math.Pow(0.85, float64(i/20000))
			jobs <- Job{encode(s), encode(reverse(s)), lr}
			if i%40000 == 0 {
				fmt.Printf("Step %d | Accuracy: %.1f%% | LR: %.4f\n", i, model.Validate(100), lr)
			}
		}
		close(jobs)
		wg.Wait()
		d, _ := json.Marshal(model)
		os.WriteFile(file, d, 0644)
	}

	test := "onionsys"
	fmt.Printf("\nTest Input: %s\nPrediction: ", test)
	displayResults(model.PredictRaw(encode(test)))
}
