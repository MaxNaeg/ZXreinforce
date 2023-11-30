
import sys
import time
import contextlib
import os
import numpy as np
import multiprocessing as mp


from copy import deepcopy
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

from .ZX_env import ZXCalculus



class VecZXCalculus():
    """This class holds a list of n_envs individual environments, 
    however, those are not executed in parallel"""
    def __init__(self, 
                 resetter_list:list,
                 n_envs:int=1,
                 max_steps:int=1000, 
                 add_reward_per_step:float=0.,
                 check_consistencty:bool=False,
                 dont_allow_stop:bool=False):
        """resetter_list: list of resetters for the environments
        n_envs: number of environments to be created
        max_steps: maximum number of steps before environament resets,
        add_reward_per_step: reward added for each step,
        check_consistencty: whether to check if the graph is consistent after each step,
        dont_allow_stop: whether to allow the stop action,
        """
        self.n_envs = n_envs
        self.env_list = [ZXCalculus(max_steps=max_steps, 
                                    add_reward_per_step=add_reward_per_step,
                                    resetter=resetter_list[idx],
                                    check_consistencty=check_consistencty,
                                    dont_allow_stop=dont_allow_stop) for idx in range(n_envs)]
        
    def step(self, actions:np.ndarray) -> tuple:
        """actions: array of actions for each environment,
        returns: tuple of lists of observations, masks, rewards and dones
        """
        
  
        results = np.array([env.step(act) for env, act in zip(self.env_list, actions)], dtype="O")
        obs = list(results[:, 0])
        mask = list(results[:, 1])
        rewards = np.array(results[:, 2], dtype=np.float32)
        done = np.array(results[:, 3], dtype=np.int32)
        return obs, mask, rewards, done
    
    def reset(self) -> list:
        """returns: list of observations and masks"""
        results = np.array([env.reset() for env in self.env_list], dtype="O")
        obs = list(results[:, 0])
        mask = list(results[:, 1])
        return obs, mask



# The following is slightly adapted from https://github.com/DLR-RM/stable-baselines3
class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorEnv():
    """Vectorized environment that runs multiple environments in parallel.
    It uses ``multiprocessing`` processes, and pipes for communication.
    """

    def __init__(
        self,
        env_fns: Sequence[callable],
        copy: bool = True,
        daemon: bool = True,
    ):
        """Vectorized environment that runs multiple environments in parallel.

        Args:
            env_fns: Functions that create the environments..
            copy: If ``True``, then the :meth:`~AsyncVectorEnv.reset` and :meth:`~AsyncVectorEnv.step` methods
                return a copy of the observations.
            daemon: If ``True``, then subprocesses have ``daemon`` flag turned on; that is, they will quit if
                the head process quits. However, ``daemon=True`` prevents subprocesses to spawn children,
                so for some environments you may want to have it set to ``False``.
        """
        # This bundles all the processes in one context
        # Use this ctx instead of mp to spwan all mp objects 
        ctx = mp.get_context()
        self.env_fns = env_fns
        self.copy = copy
        
        self.num_envs = len(env_fns)
        self.closed = False


        # TODO: Start method "spawn or forfk?"
        self.parent_pipes, self.processes = [], []
        # Queues are one way to communicate
        # process can use .put(something), queue can use .get(whats put down)
        self.error_queue = ctx.Queue()
        target = _worker
        # Seems to be needed, see docstring of function clear_mpi_env_vars.
        # TODO: check if cloudpickle needed
        with clear_mpi_env_vars():
            # Hacky way to make tensorflow use only cpu in environments:
            # TODO: Make pretty with statement
            #GPU_devices = [dev for dev in tf.config.get_visible_devices() if dev.device_type == 'GPU']
            #tf.config.set_visible_devices([], 'GPU')
            for idx, env_fn in enumerate(self.env_fns):
                # Pipes are another way to communicate with .send(some) and .recieve(some)
                parent_pipe, child_pipe = ctx.Pipe()
                # This spawns a new process where the environement runs in
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()
            # Give tensorflow its GPU back   
            #tf.config.set_visible_devices(GPU_devices, 'GPU')
        self._state = AsyncState.DEFAULT

    def reset(self):
        self._reset_async()
        return self._reset_wait()
    
    def step(self, actions):
        self._step_async(actions)
        return self._step_wait()

    def _reset_async(
        self,
    ):
        """Send calls to the :obj:`reset` methods of the sub-environments.

        To get the results of these calls, you may invoke :meth:`reset_wait`.

        Args:
            seed: List of seeds for each environment
            options: The reset option
        """
        self._assert_is_running()

        if self._state != AsyncState.DEFAULT:
            raise Exception(
                f"Calling `reset_async` while waiting for a pending call to `{self._state.value}` to complete",
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("reset", {}))
        self._state = AsyncState.WAITING_RESET

    def _reset_wait(
        self,
        timeout: Optional[Union[int, float]] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            timeout: Number of seconds before the call to `reset_wait` times out. If `None`, the call to `reset_wait` never times out.
            seed: ignored
            options: ignored

        Returns:
            A tuple of batched observations and list of dictionaries
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise Exception(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `reset_wait` has timed out after {timeout} second(s)."
            )
        
        successes = []
        observations_list, mask_list = [], []
        for i, pipe in enumerate(self.parent_pipes):
            result, success = pipe.recv()
            obs, mask = result

            successes.append(success)
            observations_list.append(obs)
            mask_list.append(mask)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT


        return (deepcopy(observations_list) if self.copy else observations_list,
                mask_list)

    def _step_async(self, actions: np.ndarray):
        """Send the calls to :obj:`step` to each sub-environment.

        Args:
            actions: Batch of actions. element of :attr:`~VectorEnv.action_space`
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise Exception(
                f"Calling `step_async` while waiting for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def _step_wait(
        self, timeout: Optional[Union[int, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Wait for the calls to :obj:`step` in each sub-environment to finish.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait` times out. If ``None``, the call to :meth:`step_wait` never times out.

        Returns:
             The batched environment step information, (obs, reward, terminated, truncated, info)
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise Exception(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `step_wait` has timed out after {timeout} second(s)."
            )

        observations_list, mask_list, rewards, dones = [], [], [], []

        successes = []
        for i, pipe in enumerate(self.parent_pipes):
            result, success = pipe.recv()
            obs, mask, rew, done = result

            successes.append(success)
            observations_list.append(obs)
            mask_list.append(mask)
            rewards.append(rew)
            dones.append(done)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return (
            deepcopy(observations_list) if self.copy else observations_list,
            mask_list,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.int32),
        )


    def close(
        self, timeout: Optional[Union[int, float]] = None, terminate: bool = False
    ):
        """Close the environments & clean up the extra resources (processes and pipes).

        Args:
            timeout: Number of seconds before the call to :meth:`close` times out. If ``None``,
                the call to :meth:`close` never times out. If the call to :meth:`close`
                times out, then all processes are terminated.
            terminate: If ``True``, then the :meth:`close` operation is forced and all processes are terminated.

        Raises:
            TimeoutError: If :meth:`close` timed out.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                function = getattr(self, f"{self._state.value}_wait")
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()
        self.closed = True

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True


    def _assert_is_running(self):
        if self.closed:
            raise Exception(
                f"Trying to operate on `{type(self).__name__}`, after a call to `close()`."
            )

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for i in range(num_errors):
            index, exctype, value = self.error_queue.get()
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

            if i == num_errors - 1:
                raise exctype(value)

    def __del__(self):
        """On deleting the object, checks that the vector environment is closed."""
        if not getattr(self, "closed", True) and hasattr(self, "_state"):
            self.close(terminate=True)




def _worker(index, env_fn, pipe, parent_pipe, error_queue):
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, mask = env.reset(**data)
                pipe.send(((observation, mask), True))

            elif command == "step":
                (
                    observation,
                    mask,
                    reward,
                    done
                ) = env.step(data)
                pipe.send(((observation, mask, reward, done), True))
            elif command == "close":
                pipe.send((None, True))
                break
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `close`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        pass
        #env.close()



@contextlib.contextmanager
def clear_mpi_env_vars():
    """Clears the MPI of environment variables.

    `from mpi4py import MPI` will call `MPI_Init` by default.
    If the child process has MPI environment variables, MPI will think that the child process
    is an MPI process just like the parent and do bad things such as hang.

    This context manager is a hacky way to clear those environment variables
    temporarily such as when we are starting multiprocessing Processes.

    Yields:
        Yields for the context manager
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ["OMPI_", "PMI_"]:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)

class CloudpickleWrapper:
    """Wrapper that uses cloudpickle to pickle and unpickle the result."""

    def __init__(self, fn: callable):
        """Cloudpickle wrapper for a function."""
        self.fn = fn

    def __getstate__(self):
        """Get the state using `cloudpickle.dumps(self.fn)`."""
        import cloudpickle

        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        """Sets the state with obs."""
        import pickle

        self.fn = pickle.loads(ob)

    def __call__(self):
        """Calls the function `self.fn` with no arguments."""
        return self.fn()