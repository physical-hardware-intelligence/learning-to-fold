# Learning to Fold on SO-101 arms

We are reproducing Ilia Larchenko's LeHome Challenge policy on our own SO-101 arms: first in the competition's simulator, then on a real two-arm rig, and then trying to improve it with our own data.

His policy is a modified π₀.₅ trained to fold tops and trousers with two arms. It placed first of 62 in the simulation round and second in the real-robot round. The paper is [Learning to Fold](https://arxiv.org/abs/2606.27163); his code is [IliaLarchenko/lehome_solution](https://github.com/IliaLarchenko/lehome_solution). This is not his repository. It holds our plan, the tools we wrote around his code, and every result, including the ones that fail. Each test's pass rule is written down before the test runs.

## Where it stands

| | |
|---|---|
| Model server | Runs on our RTX 4080 SUPER. On recorded demos it had never seen, its error was 0.385× that of holding still, and each wiring fault we planted (swapped wrist cameras, the top camera rotated 180°, another episode's images) made it reliably worse. That pass came on the second run: the first failed a bar we had set too high, and its FAIL stays in the plan. 0.28 s per answer on the box |
| Robot program | Ilia's runner, ported to macOS and LeRobot 0.6.0. Tested against the server with recorded frames, not yet with arms |
| Simulator | Isaac Sim 5.1 installs without sudo and starts headless (gate S1, 2026-10-17). Next: a smoke test, then his simulation score |
| Real rig | Waiting on a replacement motor. The first zero-shot measurement is planned for Oct 31 |

The plan, with every gate, known trap and log entry: [docs/plan.md](docs/plan.md).

## Running it

- The simulator on a GPU box: [docs/sim.md](docs/sim.md)
- The robot program on a Mac: [scripts/README.md](scripts/README.md)

## What's here

```
docs/plan.md      the plan: tracks, gates, traps, results, log
docs/sim.md       simulator runbook, every command explained
scripts/          download checks, the replay test, the Mac robot program, simulator scripts
configs/          rig config template for the robot program
```

## Credit and license

Ilia Larchenko's lehome_solution is Apache-2.0. Two of its files are adapted here with every change marked; see [scripts/LICENSE.lehome_solution](scripts/LICENSE.lehome_solution). The simulator, assets and datasets come from the LeHome Challenge organizers.

Our code is Apache-2.0. Φ (Physical Hardware Intelligence) is a student interest group at Northeastern University, Silicon Valley, not affiliated with or branded by the university. More of our work: [phi](https://github.com/physical-hardware-intelligence/phi).
