# Zhang — apple harvesting robot — speak-from notes

- **Cite:** Zhang, Lammers et al., "System design and control of an apple harvesting robot." *Mechatronics*, **2021** (ScienceDirect S0957415821001173). Preprint: arXiv **2010.11296**. Group: Zhaojian Li, Michigan State University.
- **Why:** his signature work — a specific hardware system of his to reference. The applied, real-hardware, field-manipulation end of the spectrum our learned-VLA work abstracts.

## System design (grounded)

- **Manipulator:** **3-DOF**, **hybrid pneumatic + motor actuation** for fast, dexterous reach.
- **End-effector:** **vacuum-based** gripper to detach apples.
- **Perception:** vision system with **deep-learning fruit detection**.
- Full mechatronic integration: detection → 3-DOF arm → vacuum pick.

## Field results (grounded)

- Human-in-the-loop field test: **80 of 97** apples detached → **~82% success**.
- **Average picking rate: 3.6 s / apple.**
- (Follow-on: fully automated version + field evaluation, *J. Field Robotics* 2024, arXiv 2203.00582 "Algorithm Design and Integration" — cite if discussing the automated pipeline.)

## Angle for talking to Zhang

- Position our project as the **learned-policy, tight-compute successor** to his hand-engineered harvesting control: his stack = purpose-built perception + fixed control for one task; a VLA = one general policy across tasks — *if* it fits the edge compute of a field robot. That "if" is exactly our LoRA+PTQ contribution.
- Concrete hook: his 3.6 s/apple cycle is a real-world latency budget. Frames why our **latency metric** (§7) matters — a field manipulator has a hard per-action time budget, and quantization is how you hit it on Jetson-class hardware.
- Shared vocabulary: success rate + cycle time are *his* metrics too → our five-metric table speaks his language.
