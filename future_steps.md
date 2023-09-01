# ACT

# Current state

## Clean

- Remove not used files.
- Clean global functions.
- Move build logic to act_builder.

## Single Arm

- From 14DOF to 7DOF.

# Next steps

## Dataset

- Improve the dataset creation ? - Instead of 1 episode 1 chunk lets use more data.
- Adapt from hdf5 to h5.

## Clean

- Clean imitate_episode logic.
- Remove tmp folder (sim folder with act data)

## Multi-Task

- Add instruction + language encoder.
- Add FiLM to conditionate image embeddings on text embeddings.
(MT-ACT Replication)

## Training on Kitchen env

- Script to run kitchen with specific policy.
- Prepare dataset (<https://drive.google.com/drive/folders/1a-q6TpskJD3J7G2FcJBzYRb7UxG1N_K4>).
- Training.
