class AdvancedStateMachine:
    def __init__(self):
        self.current_state = "INIT"
        self.error_state = "ERROR"
        self.completed_state = "COMPLETED"
        self.valid_states = {"INIT", "RUNNING", "PROCESSING", "COMPLETED", "ERROR"}

    def transition(self, new_state):
        if new_state not in self.valid_states:
            print(f"Invalid state transition to {new_state}. Entering error state.")
            self.current_state = self.error_state
        else:
            print(f"Transitioning from {self.current_state} to {new_state}")
            self.current_state = new_state

    def get_state(self):
        return self.current_state

    def is_completed(self):
        return self.current_state == self.completed_state

    def is_in_error(self):
        return self.current_state == self.error_state


class EnhancedDecisionEngine:
    def __init__(self, input_spec, output_spec):
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.decision_map = {
            "START": "RUNNING",
            "PROCESS": "PROCESSING",
            "FINISH": "COMPLETED"
        }

    def evaluate(self, input_data):
        # Add complex rules, conditions, or even AI-driven logic here
        if input_data in self.decision_map:
            return self.decision_map[input_data]
        else:
            return "ERROR"

    def generate_response(self, decision):
        if decision == "ERROR":
            return {"status": "Failed", "message": "An error occurred during decision-making."}
        else:
            return {"status": "Success", "next_state": decision}


class FlexibleTaskExecutor:
    def __init__(self, decision_engine):
        self.decision_engine = decision_engine
        self.retry_limit = 3

    def perform_task(self, input_data):
        # Evaluate the task and retry on failure
        for attempt in range(self.retry_limit):
            print(f"Task attempt {attempt + 1}: Processing input {input_data}")
            decision = self.decision_engine.evaluate(input_data)
            response = self.decision_engine.generate_response(decision)

            if decision == "ERROR":
                print(f"Error encountered on attempt {attempt + 1}. Retrying...")
                continue
            else:
                print(f"Task succeeded with response: {response}")
                return response

        print("Max retry limit reached. Task failed.")
        return {"status": "Failed", "message": "Max retry limit reached."}


class AgentXFramework:
    def __init__(self, architecture, input_spec, output_spec):
        self.state_machine = AdvancedStateMachine()
        self.decision_engine = EnhancedDecisionEngine(input_spec, output_spec)
        self.task_executor = FlexibleTaskExecutor(self.decision_engine)
        self.architecture = architecture
        self.memory = {}

    def run_agent(self, input_data):
        print(f"Running agent with input: {input_data}")
        self.state_machine.transition("RUNNING")

        # Memory handling (optional): Store some context
        self.memory["last_input"] = input_data

        task_result = self.task_executor.perform_task(input_data)
        if task_result["status"] == "Failed":
            self.state_machine.transition("ERROR")
        else:
            self.state_machine.transition(task_result["next_state"])

        return task_result

    def get_memory(self):
        return self.memory
