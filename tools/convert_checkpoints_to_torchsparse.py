import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_before", metavar="FILE", help="Original checkpoint.")
    parser.add_argument("ckpt_after", metavar="FILE", help="Converted checkpoint.")
    args, opts = parser.parse_known_args()

    ckpt_before = args.ckpt_before
    ckpt_after = args.ckpt_after

    cp_old = torch.load(ckpt_before, map_location="cpu")
    model = cp_old["state_dict"]
    new_model = dict()

    for key in model:
        if key.startswith("encoders.lidar.backbone") and ".bn." not in key:
            is_sparseconv_weight = len(model[key].shape) > 1
        else:
            is_sparseconv_weight = False
        if is_sparseconv_weight:
            new_key = key.replace(".weight", ".kernel")
            weights = model[key]

            kx, ky, kz, ic, oc = weights.shape
            converted_weights = weights.reshape(-1, ic, oc)
            if converted_weights.shape[0] == 1:
                converted_weights = converted_weights[0]

            elif converted_weights.shape[0] == 27:
                offsets = [list(range(kz)), list(range(ky)), list(range(kx))]
                kykx = ky * kx
                offsets = [
                    (x * kykx + y * kx + z)
                    for z in offsets[0]
                    for y in offsets[1]
                    for x in offsets[2]
                ]
                offsets = torch.tensor(
                    offsets, dtype=torch.int64, device=converted_weights.device
                )
                converted_weights = converted_weights[offsets]

        else:
            new_key = key
            converted_weights = model[key]
        new_model[new_key] = converted_weights

    cp_old["state_dict"] = new_model
    torch.save(cp_old, ckpt_after)


if __name__ == "__main__":
    main()
