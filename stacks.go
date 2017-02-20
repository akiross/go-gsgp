package main

type stackf []float64

func (self *stackf) push(v float64) {
	*self = append(*self, v)
}
func (self *stackf) pop() float64 {
	v := self.top()
	*self = (*self)[:len(*self)-1]
	return v
}
func (self *stackf) top() float64 {
	return (*self)[len(*self)-1]
}

type stackn []*Node

func (self *stackn) push(v *Node) {
	*self = append(*self, v)
}
func (self *stackn) pop() *Node {
	v := self.top()
	*self = (*self)[:len(*self)-1]
	return v
}
func (self *stackn) top() *Node {
	return (*self)[len(*self)-1]
}

type stackb []bool

func (self *stackb) push(v bool) {
	*self = append(*self, v)
}
func (self *stackb) pop() bool {
	v := self.top()
	*self = (*self)[:len(*self)-1]
	return v
}
func (self *stackb) top() bool {
	return (*self)[len(*self)-1]
}
