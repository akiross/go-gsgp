// Code generated by protoc-gen-go. DO NOT EDIT.
// source: evolution.proto

/*
Package pb is a generated protocol buffer package.

It is generated from these files:
	evolution.proto

It has these top-level messages:
	Node
	RandomTree
	Individual
	Population
	Evolution
*/
package pb

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion2 // please upgrade the proto package

type Node_Symbol int32

const (
	Node_ADD   Node_Symbol = 0
	Node_SUB   Node_Symbol = 1
	Node_MUL   Node_Symbol = 2
	Node_DIV   Node_Symbol = 3
	Node_CONST Node_Symbol = 4
	Node_VAR   Node_Symbol = 5
)

var Node_Symbol_name = map[int32]string{
	0: "ADD",
	1: "SUB",
	2: "MUL",
	3: "DIV",
	4: "CONST",
	5: "VAR",
}
var Node_Symbol_value = map[string]int32{
	"ADD":   0,
	"SUB":   1,
	"MUL":   2,
	"DIV":   3,
	"CONST": 4,
	"VAR":   5,
}

func (x Node_Symbol) String() string {
	return proto.EnumName(Node_Symbol_name, int32(x))
}
func (Node_Symbol) EnumDescriptor() ([]byte, []int) { return fileDescriptor0, []int{0, 0} }

type Individual_Operator int32

const (
	Individual_INIT Individual_Operator = 0
	Individual_XO   Individual_Operator = 1
	Individual_MUT  Individual_Operator = 2
	Individual_REPR Individual_Operator = 3
)

var Individual_Operator_name = map[int32]string{
	0: "INIT",
	1: "XO",
	2: "MUT",
	3: "REPR",
}
var Individual_Operator_value = map[string]int32{
	"INIT": 0,
	"XO":   1,
	"MUT":  2,
	"REPR": 3,
}

func (x Individual_Operator) String() string {
	return proto.EnumName(Individual_Operator_name, int32(x))
}
func (Individual_Operator) EnumDescriptor() ([]byte, []int) { return fileDescriptor0, []int{2, 0} }

type Node struct {
	Symbol   Node_Symbol `protobuf:"varint,1,opt,name=symbol,enum=pb.Node_Symbol" json:"symbol,omitempty"`
	Parent   *Node       `protobuf:"bytes,2,opt,name=parent" json:"parent,omitempty"`
	Children []*Node     `protobuf:"bytes,3,rep,name=children" json:"children,omitempty"`
}

func (m *Node) Reset()                    { *m = Node{} }
func (m *Node) String() string            { return proto.CompactTextString(m) }
func (*Node) ProtoMessage()               {}
func (*Node) Descriptor() ([]byte, []int) { return fileDescriptor0, []int{0} }

func (m *Node) GetSymbol() Node_Symbol {
	if m != nil {
		return m.Symbol
	}
	return Node_ADD
}

func (m *Node) GetParent() *Node {
	if m != nil {
		return m.Parent
	}
	return nil
}

func (m *Node) GetChildren() []*Node {
	if m != nil {
		return m.Children
	}
	return nil
}

type RandomTree struct {
	Data          []int32   `protobuf:"varint,1,rep,packed,name=data" json:"data,omitempty"`
	SemanticTrain []float64 `protobuf:"fixed64,2,rep,packed,name=semantic_train,json=semanticTrain" json:"semantic_train,omitempty"`
	SemanticTest  []float64 `protobuf:"fixed64,3,rep,packed,name=semantic_test,json=semanticTest" json:"semantic_test,omitempty"`
}

func (m *RandomTree) Reset()                    { *m = RandomTree{} }
func (m *RandomTree) String() string            { return proto.CompactTextString(m) }
func (*RandomTree) ProtoMessage()               {}
func (*RandomTree) Descriptor() ([]byte, []int) { return fileDescriptor0, []int{1} }

func (m *RandomTree) GetData() []int32 {
	if m != nil {
		return m.Data
	}
	return nil
}

func (m *RandomTree) GetSemanticTrain() []float64 {
	if m != nil {
		return m.SemanticTrain
	}
	return nil
}

func (m *RandomTree) GetSemanticTest() []float64 {
	if m != nil {
		return m.SemanticTest
	}
	return nil
}

type Individual struct {
	Op            Individual_Operator `protobuf:"varint,1,opt,name=op,enum=pb.Individual_Operator" json:"op,omitempty"`
	FitnessTrain  float64             `protobuf:"fixed64,2,opt,name=fitness_train,json=fitnessTrain" json:"fitness_train,omitempty"`
	FitnessTest   float64             `protobuf:"fixed64,3,opt,name=fitness_test,json=fitnessTest" json:"fitness_test,omitempty"`
	SemanticTrain []float64           `protobuf:"fixed64,4,rep,packed,name=semantic_train,json=semanticTrain" json:"semantic_train,omitempty"`
	SemanticTest  []float64           `protobuf:"fixed64,5,rep,packed,name=semantic_test,json=semanticTest" json:"semantic_test,omitempty"`
	Contrib       []byte              `protobuf:"bytes,6,opt,name=contrib,proto3" json:"contrib,omitempty"`
	Rt            []*RandomTree       `protobuf:"bytes,7,rep,name=rt" json:"rt,omitempty"`
}

func (m *Individual) Reset()                    { *m = Individual{} }
func (m *Individual) String() string            { return proto.CompactTextString(m) }
func (*Individual) ProtoMessage()               {}
func (*Individual) Descriptor() ([]byte, []int) { return fileDescriptor0, []int{2} }

func (m *Individual) GetOp() Individual_Operator {
	if m != nil {
		return m.Op
	}
	return Individual_INIT
}

func (m *Individual) GetFitnessTrain() float64 {
	if m != nil {
		return m.FitnessTrain
	}
	return 0
}

func (m *Individual) GetFitnessTest() float64 {
	if m != nil {
		return m.FitnessTest
	}
	return 0
}

func (m *Individual) GetSemanticTrain() []float64 {
	if m != nil {
		return m.SemanticTrain
	}
	return nil
}

func (m *Individual) GetSemanticTest() []float64 {
	if m != nil {
		return m.SemanticTest
	}
	return nil
}

func (m *Individual) GetContrib() []byte {
	if m != nil {
		return m.Contrib
	}
	return nil
}

func (m *Individual) GetRt() []*RandomTree {
	if m != nil {
		return m.Rt
	}
	return nil
}

type Population struct {
	Generation  int32         `protobuf:"varint,1,opt,name=generation" json:"generation,omitempty"`
	Individuals []*Individual `protobuf:"bytes,2,rep,name=individuals" json:"individuals,omitempty"`
}

func (m *Population) Reset()                    { *m = Population{} }
func (m *Population) String() string            { return proto.CompactTextString(m) }
func (*Population) ProtoMessage()               {}
func (*Population) Descriptor() ([]byte, []int) { return fileDescriptor0, []int{3} }

func (m *Population) GetGeneration() int32 {
	if m != nil {
		return m.Generation
	}
	return 0
}

func (m *Population) GetIndividuals() []*Individual {
	if m != nil {
		return m.Individuals
	}
	return nil
}

type Evolution struct {
	// Evolutionary process consists in a set of generations
	Generations []*Population `protobuf:"bytes,1,rep,name=generations" json:"generations,omitempty"`
}

func (m *Evolution) Reset()                    { *m = Evolution{} }
func (m *Evolution) String() string            { return proto.CompactTextString(m) }
func (*Evolution) ProtoMessage()               {}
func (*Evolution) Descriptor() ([]byte, []int) { return fileDescriptor0, []int{4} }

func (m *Evolution) GetGenerations() []*Population {
	if m != nil {
		return m.Generations
	}
	return nil
}

func init() {
	proto.RegisterType((*Node)(nil), "pb.Node")
	proto.RegisterType((*RandomTree)(nil), "pb.RandomTree")
	proto.RegisterType((*Individual)(nil), "pb.Individual")
	proto.RegisterType((*Population)(nil), "pb.Population")
	proto.RegisterType((*Evolution)(nil), "pb.Evolution")
	proto.RegisterEnum("pb.Node_Symbol", Node_Symbol_name, Node_Symbol_value)
	proto.RegisterEnum("pb.Individual_Operator", Individual_Operator_name, Individual_Operator_value)
}

func init() { proto.RegisterFile("evolution.proto", fileDescriptor0) }

var fileDescriptor0 = []byte{
	// 453 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x8c, 0x53, 0x4d, 0x6f, 0xd3, 0x40,
	0x10, 0xed, 0xae, 0x3f, 0x9a, 0x4e, 0xd2, 0xd4, 0xda, 0x0b, 0x3e, 0x55, 0xc6, 0x80, 0xea, 0x93,
	0x41, 0xe5, 0x8c, 0x44, 0x21, 0x3d, 0x44, 0x82, 0xa4, 0xda, 0xb8, 0x15, 0x27, 0xd0, 0x3a, 0x5e,
	0x60, 0x25, 0x67, 0xd7, 0xda, 0xdd, 0x54, 0xe2, 0x7f, 0x71, 0xe1, 0xdf, 0xa1, 0x75, 0xfc, 0x11,
	0x04, 0x07, 0x6e, 0xe3, 0x37, 0x6f, 0x47, 0xef, 0xbd, 0x19, 0xc3, 0x05, 0x7f, 0x54, 0xf5, 0xde,
	0x0a, 0x25, 0xf3, 0x46, 0x2b, 0xab, 0x08, 0x6e, 0xca, 0xf4, 0x17, 0x02, 0x7f, 0xa5, 0x2a, 0x4e,
	0xae, 0x20, 0x34, 0x3f, 0x76, 0xa5, 0xaa, 0x63, 0x94, 0xa0, 0x6c, 0x7e, 0x7d, 0x91, 0x37, 0x65,
	0xee, 0x3a, 0xf9, 0xa6, 0x85, 0x69, 0xd7, 0x26, 0x09, 0x84, 0x0d, 0xd3, 0x5c, 0xda, 0x18, 0x27,
	0x28, 0x9b, 0x5e, 0x4f, 0x7a, 0x22, 0xed, 0x70, 0xf2, 0x1c, 0x26, 0xdb, 0xef, 0xa2, 0xae, 0x34,
	0x97, 0xb1, 0x97, 0x78, 0x7f, 0x70, 0x86, 0x4e, 0xfa, 0x16, 0xc2, 0xc3, 0x64, 0x72, 0x0a, 0xde,
	0xcd, 0x62, 0x11, 0x9d, 0xb8, 0x62, 0x73, 0xff, 0x2e, 0x42, 0xae, 0xf8, 0x78, 0xff, 0x21, 0xc2,
	0xae, 0x58, 0x2c, 0x1f, 0x22, 0x8f, 0x9c, 0x41, 0xf0, 0x7e, 0xbd, 0xda, 0x14, 0x91, 0xef, 0xb0,
	0x87, 0x1b, 0x1a, 0x05, 0x69, 0x0d, 0x40, 0x99, 0xac, 0xd4, 0xae, 0xd0, 0x9c, 0x13, 0x02, 0x7e,
	0xc5, 0x2c, 0x8b, 0x51, 0xe2, 0x65, 0x01, 0x6d, 0x6b, 0xf2, 0x02, 0xe6, 0x86, 0xef, 0x98, 0xb4,
	0x62, 0xfb, 0xc5, 0x6a, 0x26, 0x64, 0x8c, 0x13, 0x2f, 0x43, 0xf4, 0xbc, 0x47, 0x0b, 0x07, 0x92,
	0x67, 0x70, 0x3e, 0xd2, 0xb8, 0xb1, 0xad, 0x6a, 0x44, 0x67, 0x03, 0x8b, 0x1b, 0x9b, 0xfe, 0xc4,
	0x00, 0x4b, 0x59, 0x89, 0x47, 0x51, 0xed, 0x59, 0x4d, 0xae, 0x00, 0xab, 0xa6, 0xcb, 0xea, 0x89,
	0xb3, 0x37, 0xf6, 0xf2, 0x75, 0xc3, 0x35, 0xb3, 0x4a, 0x53, 0xac, 0x1a, 0x37, 0xfc, 0xab, 0xb0,
	0x92, 0x1b, 0x33, 0x48, 0x40, 0x6e, 0x78, 0x07, 0x1e, 0x14, 0x3c, 0x85, 0xd9, 0x40, 0x3a, 0x08,
	0x70, 0x9c, 0x69, 0xcf, 0xe1, 0xc6, 0xfe, 0xc3, 0x8b, 0xff, 0x5f, 0x5e, 0x82, 0xbf, 0xbd, 0x90,
	0x18, 0x4e, 0xb7, 0x4a, 0x5a, 0x2d, 0xca, 0x38, 0x4c, 0x50, 0x36, 0xa3, 0xfd, 0x27, 0xb9, 0x04,
	0xac, 0x6d, 0x7c, 0xda, 0x6e, 0x6d, 0xee, 0x6c, 0x8d, 0x09, 0x53, 0xac, 0x6d, 0xfa, 0x12, 0x26,
	0xbd, 0x3b, 0x32, 0x01, 0x7f, 0xb9, 0x5a, 0x16, 0xd1, 0x09, 0x09, 0x01, 0x7f, 0x5a, 0xf7, 0x7b,
	0x2b, 0x22, 0xec, 0x5a, 0xf4, 0xf6, 0x8e, 0x46, 0x5e, 0xfa, 0x19, 0xe0, 0x4e, 0x35, 0xfb, 0x9a,
	0xb9, 0xc3, 0x23, 0x97, 0x00, 0xdf, 0xb8, 0x74, 0xef, 0x85, 0x92, 0x6d, 0x7a, 0x01, 0x3d, 0x42,
	0xc8, 0x2b, 0x98, 0x8a, 0x21, 0x47, 0xd3, 0x6e, 0xab, 0xd3, 0x31, 0xc6, 0x4b, 0x8f, 0x29, 0xe9,
	0x1b, 0x38, 0xbb, 0xed, 0xef, 0xda, 0x3d, 0x1f, 0x87, 0x99, 0xf6, 0x14, 0xba, 0xe7, 0xa3, 0x06,
	0x7a, 0x4c, 0x29, 0xc3, 0xf6, 0x57, 0x78, 0xfd, 0x3b, 0x00, 0x00, 0xff, 0xff, 0x97, 0x1f, 0x9a,
	0x3f, 0x1d, 0x03, 0x00, 0x00,
}
