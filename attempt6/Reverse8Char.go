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
	"time"
)

type Neuron struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

func sigmoid(x float64) float64           { return 1.0 / (1.0 + math.Exp(-x)) }
func sigmoidDerivative(s float64) float64  { return s * (1.0 - s) }

func (n *Neuron) Process(inputs []float64) float64 {
	var sum float64
	for i, v := range inputs {
		sum += v * n.Weights[i]
	}
	return sigmoid(sum + n.Bias)
}

type MLP struct {
	Hidden []*Neuron `json:"hidden"`
	Output []*Neuron `json:"output"`
}

func NewMLP(inputDim, hiddenDim, outputDim int) *MLP {
	m := &MLP{
		Hidden: make([]*Neuron, hiddenDim),
		Output: make([]*Neuron, outputDim),
	}
	for i := range m.Hidden {
		m.Hidden[i] = &Neuron{Weights: randomWeights(inputDim), Bias: rand.Float64()}
	}
	for i := range m.Output {
		m.Output[i] = &Neuron{Weights: randomWeights(hiddenDim), Bias: rand.Float64()}
	}
	return m
}

func randomWeights(n int) []float64 {
	w := make([]float64, n)
	for i := range w { w[i] = rand.NormFloat64() * 0.1 }
	return w
}

func (m *MLP) Train(inputs []float64, targets []float64, lr float64) {
	// 1. Forward Pass
	hOutputs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { hOutputs[i] = n.Process(inputs) }

	oOutputs := make([]float64, len(m.Output))
	for i, n := range m.Output { oOutputs[i] = n.Process(hOutputs) }

	// 2. Backprop Output Layer
	oGradients := make([]float64, len(m.Output))
	for i := range m.Output {
		err := targets[i] - oOutputs[i]
		oGradients[i] = err * sigmoidDerivative(oOutputs[i])
	}

	// 3. Backprop Hidden Layer
	hGradients := make([]float64, len(m.Hidden))
	for i := range m.Hidden {
		var err float64
		for j := range m.Output {
			err += oGradients[j] * m.Output[j].Weights[i]
		}
		hGradients[i] = err * sigmoidDerivative(hOutputs[i])
	}

	// 4. Update Weights
	for i, n := range m.Output {
		for j := range n.Weights { n.Weights[j] += hOutputs[j] * oGradients[i] * lr }
		n.Bias += oGradients[i] * lr
	}
	for i, n := range m.Hidden {
		for j := range n.Weights { n.Weights[j] += inputs[j] * hGradients[i] * lr }
		n.Bias += hGradients[i] * lr
	}
}

func (m *MLP) Predict(inputs []float64) string {
	hOutputs := make([]float64, len(m.Hidden))
	for i, n := range m.Hidden { hOutputs[i] = n.Process(inputs) }
	
	res := ""
	for _, n := range m.Output {
		val := n.Process(hOutputs)
		res += string(rune(int(val*255.0 + 0.5)))
	}
	return res
}

func stringToFloats(s string) []float64 {
	f := make([]float64, 8)
	for i := 0; i < 8 && i < len(s); i++ { f[i] = float64(s[i]) / 255.0 }
	return f
}

func reverse(s string) string {
	r := []rune(s); for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 { r[i], r[j] = r[j], r[i] }; return string(r)
}

func generateRandomString(n int) string {
	const l = "abcdefghijklmnopqrstuvwxyz"; b := make([]byte, n)
	for i := range b { b[i] = l[rand.Intn(len(l))] }; return string(b)
}

func main() {
	rand.Seed(time.Now().UnixNano())
	const file = "deep_weights.json"
	var model *MLP

	if _, err := os.Stat(file); err == nil {
		fmt.Println("Loading Deep Model...")
		data, _ := os.ReadFile(file)
		json.Unmarshal(data, &model)
	} else {
		fmt.Println("Training Deep Model (Generalization)...")
		// @BUG: model = NewMLP(8, 16, 8)
		model = NewMLP(8, 64, 8)
		// @BUG: for e := 0; e < 200000; e++ {
		for e := 0; e < 400000; e++ {
			s := generateRandomString(8)
			targets := stringToFloats(reverse(s))
			model.Train(stringToFloats(s), targets, 0.2)
			if e%20000 == 0 { fmt.Printf("Epoch %d/200000\n", e) }
		}
		d, _ := json.Marshal(model)
		os.WriteFile(file, d, 0644)
	}

	test := "onionsys"
	fmt.Printf("\nInput:  %s\nTarget: %s\nOutput: %s\n", test, reverse(test), model.Predict(stringToFloats(test)))
}
