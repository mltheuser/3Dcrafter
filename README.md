# 3Dcrafter

You load in some 3D model and the program will try to recreate it as faithful as possible using only minecraft blocks. Then you export to a schema file so you can rebuild it in the actual game.

## Example

#### Load a subset of blocks
![block-palette](https://github.com/mltheuser/3Dcrafter/assets/25958978/8cb8ef41-d141-4b84-ba10-de71514d679d)

#### Start the iterative process of training the block distribution using differential rendering (gumble softmax trick)
![train_start](https://github.com/mltheuser/3Dcrafter/assets/25958978/ce055812-66ab-4246-9f3b-6c9ca833ff60)

![train_end](https://github.com/mltheuser/3Dcrafter/assets/25958978/e01500f3-d452-412e-9b40-5d25a4925e15)

#### Export the learned blocks to a [litematica](https://github.com/maruohon/litematica) schema
<img width="1710" alt="Screenshot 2024-06-19 at 08 44 24" src="https://github.com/mltheuser/3Dcrafter/assets/25958978/e5246225-e148-4c54-a518-fc9137b740c5">

`
Note: There is a clear discrepancy between the training renderer and the ingame render (occlusion and lighting). Bridging this gap however will need a fundamentaly different approach to the current one. Maybe something for a second iteration of this project^^.
`

### Minecraft Importer

We load blocks directly from the games asset folder and convert them from their blockstate representation [https://minecraft.fandom.com/wiki/Tutorials/Models] into trinagulated textured meshes.

![loaded_assets](https://github.com/mltheuser/3Dcrafter/assets/25958978/df27980a-8810-41e9-8fe8-1be5e228e712)

