#pragma once
#include <iostream>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <cmath>

#define log(X) { std::cout << X << std::endl;}
#define OPR_ADD '+'
#define OPR_SUB  '-'
#define OPR_MUL '*'
#define OPR_DIV '/'


class DCG;
class Item;


class DAG{
public:


	struct Node{
		bool is_operator;
		Item* operand_1;
		Item* operand_2;
		Item* item;
		double value;
		char operation;
		Node* prev;


		Node(Item* op_1, Item* op_2, Item* item, const double& value, const char& operation, Node* prev){
			this->operand_1 = op_1;
			this->operand_2 = op_2;
			this->item = item;
			this->value = value;
			this->operation = operation;
			this->prev = prev;
		}
	};


	Node* result;

public:
	DCG(){
		this->result = nullptr;
	}
	
	Node* add_item(const double& value,
				  const char& operation,
	      		          Item* operand_1,
				  Item* operand_2,
				  Item* res){


		if(!result){
			result = new Node(operand_1, operand_2, res, value, operation, nullptr);
			return result;
		}

		Node* newNode = new Node(operand_1, operand_2, res, value, operation, result);
		result = newNode;
		
		return result;
	}


};


class Item{
public:
	double value;
	bool requires_grad;
	double grad;
	static DCG dc_graph; // Dynamic Computation Graph
	DCG::Node* dc_node;


public:
	
	Item(float value, bool requires_grad = false){
		this->value = value;
		this->grad = 0;
		this->requires_grad = requires_grad;
	}


	Item& operator + (Item& other){
		int operation_result = value + other.value;
		bool grad_requirement = this->requires_grad || other.requires_grad;
		Item* result = new Item(operation_result, requires_grad || other.requires_grad);
		
		if(grad_requirement){
			result->dc_node = dc_graph.add_item(operation_result, OPR_ADD, this, &other, result);
		}
		
		return *result;
	}

	
	Item& operator - (Item& other){
		int operation_result = value - other.value;
		bool grad_requirement = this->requires_grad || other.requires_grad;
		Item* result = new Item(operation_result, requires_grad || other.requires_grad);
		
		if(grad_requirement){
			result->dc_node = dc_graph.add_item(operation_result, OPR_SUB, this, &other, result);
		}
		
		return *result;
	}


	Item& operator * (Item& other){
		int operation_result = value * other.value;
		bool grad_requirement = this->requires_grad || other.requires_grad;
		Item* result = new Item(operation_result, requires_grad || other.requires_grad);
		
		if(grad_requirement){
			result->dc_node = dc_graph.add_item(operation_result, OPR_MUL, this, &other, result);
		}

		return *result;
	}


	Item& operator / (Item& other){
		int operation_result = value / other.value;
		bool grad_requirement = this->requires_grad || other.requires_grad;
		Item* result = new Item(operation_result, requires_grad || other.requires_grad);
		
		if(grad_requirement){
			result->dc_node = dc_graph.add_item(operation_result, OPR_DIV, this, &other, result);
		}
		
		return *result;
	}

	friend std::ostream& operator << (std::ostream& os, Item& item){
		os << item.value << "\n";
		return os;
	}


	void backward(){
		if(!this->dc_node) return;
		this->dc_node->item->grad = 1.0; // Outer most node from which backprop is carried out,  say 'C', ∂C/∂C = 1
		this->backpropagate(this->dc_node);
	}

	void reset_grads(double new_grad = 0){
		this->grad = new_grad;
		this->update_grads(this->da_node, new_grad);
	}

private:

	double add_backwards(Item* a, Item* b){
		return 1.0;
	}

	double sub_backwards(Item* a, Item* b){
		return 1.0;
	}

	// wrt a
	double mul_backwards(Item* a, Item* b){
		return b->value;
	}

	// wrt a for a/b
	double div_backwards_0(Item* a, Item* b){
		return 1 / b->value;
	}

	// wrt b for a/b
	double div_backwards_1(Item* a, Item* b){
		double a_ = a->value, b_ = b->value;
		return - a_ / pow(b_, 2);
	}

	std::pair<double, double>  compute_grad(const char& operation, Item* a, Item* b){
		

		switch(operation){
			case '+':
				log("<Add_Backward>");
				return {add_backwards(a, b), add_backwards(b, a)};

			case '-':
				log("<Sub_Backward>");
				return {sub_backwards(a, b), -1*sub_backwards(b, a)};

			case '*':
				log("<Mul_Backward>");
				return {mul_backwards(a, b), mul_backwards(b, a)};

			case '/':	
				log("<Div_Backward>");
				return {div_backwards_0(a, b), div_backwards_1(a, b)};
		}
	
		return {0.0, 0.0};

	}

	void backpropagate(DCG::Node* node){
		if(!node) return;

		/*
		printf("%lf (%p) %c %lf (%p) = %lf (%p), \t grad : %lf\n", node->operand_1->value, node->operand_1, node->operation,  node->operand_2->value, node->operand_2, node->value, node->item, node->item->grad); 	
		sleep(1);
		*/

		std::pair<double, double> grads = this->compute_grad(node->operation, node->operand_1, node->operand_2);

		printf("Grads : %lf   %lf\n\n", grads.first, grads.second);

		node->operand_1->grad += node->item->grad * grads.first;
		node->operand_2->grad += node->item->grad * grads.second;
		backpropagate(node->prev);
	
	}


	void update_grads(DCG::Node* node, double new_grad = 0){
		if(!node) return;

		node->operand_1->grad = new_grad;
		node->operand_2->grad = new_grad;

		update_grads(node->prev, new_grad);
	}


};


DCG Item::dc_graph; // Definition outside of the class to allocate memory for the static member
