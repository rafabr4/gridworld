import abc


class Environment(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "initialize")
            and callable(subclass.initialize)
            and hasattr(subclass, "get_all_possible_states")
            and callable(subclass.get_all_possible_states)
            and hasattr(subclass, "get_possible_actions")
            and callable(subclass.get_possible_actions)
            and hasattr(subclass, "take_action")
            and callable(subclass.take_action)
            or NotImplemented
        )

    @abc.abstractmethod
    def initialize(self, method, state):
        """Initialize environment"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_possible_states(self) -> list:
        """Get all possible states in the environment"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_possible_actions(self, state=None) -> list:
        """Get all possible actions of the current state or of a specific one"""
        raise NotImplementedError

    @abc.abstractmethod
    def take_action(self) -> tuple:
        """Sends action to the environment and receives new state and reward"""
        raise NotImplementedError
