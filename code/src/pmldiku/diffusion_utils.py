import matplotlib.pyplot as plt
import torch

from pmldiku import diffusion


def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()


def generate_new_images(
    model: diffusion.LightningDiffusion,
    n_samples=16,
    device=None,
    c=1,
    h=28,
    w=28,
):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""

    with torch.no_grad():
        if device is None:
            device = model.device
            model.diffusion_params.set_params_device(device)

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for t in list(range(model.diffusion_params.n_steps))[::-1]:
            if t % 50 == 0:
                print(f"t = {t}")
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            print(time_tensor.device, x.device)
            eta_theta = model.diffusion.backward(x, time_tensor)
            alpha_t = model.diffusion_params.alphas[t]
            alpha_t_bar = model.diffusion_params.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = model.diffusion_params.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z
    return x
