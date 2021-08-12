import torch
import torch.nn.functional as F


def pairLoss(fea1, fea2, mask):
    # fea1_size (bs, max_len, dim)
    # fea2_size (bs, max_len, dim)
    # mask_size (bs, max_len)
    # '-Inf' for padded item, '0' for others

    fea1 = F.normalize(fea1, p=2, dim=-1)
    fea2 = F.normalize(fea2, p=2, dim=-1)
    fea_sim = (fea1 * fea2).sum(dim=-1)  # (bs, max_len)
    fea_sim = torch.masked_select(fea_sim, mask == 0)
    loss = 1.0 - torch.mean(fea_sim)
    return loss


def SimpTripLoss(fea1, fea2):
    # img fea1_size (bs, max_len1, dim)  mask1_size (bs, max_len1)
    # text fea2_size (bs, max_len2, dim)  mask2_size (bs, max_len2)
    # '-Inf' for padded item, '0' for others
    # fea1 = fea1.mean(dim=1)  #(bs, dim)
    # mask2 = torch.where(mask2==0, torch.tensor([1.0],device=mask2.device), torch.tensor([0.0],device=mask2.device))
    # fea2 = (fea2 * mask2.unsqueeze(-1)).sum(dim=1) / mask2.sum(dim=1).unsqueeze(-1) #(bs, dim)

    fea1 = F.normalize(fea1, p=2, dim=-1)
    fea2 = F.normalize(fea2, p=2, dim=-1)

    # match fea1 to fea2
    sim_pos1 = (fea1 * fea2).sum(dim=1)  # (bs)
    # (bs, 1, dim)  (1, bs, dim)
    sim_neg1_all = (fea1.unsqueeze(1) * fea2.unsqueeze(0)).sum(dim=-1)  # (bs,bs)
    unmask = torch.eye(sim_pos1.size(0), dtype=torch.float32, device=sim_pos1.device)
    unmask = torch.where(unmask == 1, torch.tensor([float('-Inf')], device=unmask.device), unmask)

    sim_neg1, _ = torch.max(sim_neg1_all + unmask, 1)
    loss1 = -sim_pos1 + sim_neg1 + 0.2
    loss1 = torch.maximum(loss1, torch.zeros_like(loss1)).mean()

    # match fea2 to fea1
    sim_pos2 = (fea2 * fea1).sum(
        dim=1)  # (bs)    sim_neg2_all = (fea2.unsqueeze(1) * fea1.unsqueeze(0)).sum(dim=-1)  #(bs,bs)
    sim_neg2_all = (fea2.unsqueeze(1) * fea1.unsqueeze(0)).sum(dim=-1)  # (bs,bs)

    sim_neg2, _ = torch.max(sim_neg2_all + unmask, 1)
    loss2 = -sim_pos2 + sim_neg2 + 0.2
    loss2 = torch.maximum(loss2, torch.zeros_like(loss2)).mean()

    loss = loss1 + loss2
    return loss


def NCELoss(fea1, fea2):
    # img fea1_size (bs, max_len1, dim)  mask1_size (bs, max_len1)
    # text fea2_size (bs, max_len2, dim)  mask2_size (bs, max_len2)
    # '-Inf' for padded item, '0' for others
    # fea1 = fea1.mean(dim=1)  #(bs, dim)
    # mask2 = torch.where(mask2==0, torch.tensor([1.0],device=mask2.device), torch.tensor([0.0],device=mask2.device))
    # fea2 = (fea2 * mask2.unsqueeze(-1)).sum(dim=1) / mask2.sum(dim=1).unsqueeze(-1) #(bs, dim)

    fea1 = F.normalize(fea1, p=2, dim=-1)
    fea2 = F.normalize(fea2, p=2, dim=-1)

    # match fea1 to fea2
    sim_pos1 = (fea1 * fea2).sum(dim=1).unsqueeze(-1)  # (bs,1)
    BS = sim_pos1.size(0)
    # (bs, 1, dim)  (1, bs, dim)
    sim_neg1_all = (fea1.unsqueeze(1) * fea2.unsqueeze(0)).sum(dim=-1)  # (bs,bs)
    unmask = torch.eye(sim_pos1.size(0), dtype=torch.float32, device=sim_pos1.device)
    sim_neg1_all = torch.masked_select(sim_neg1_all, unmask == 0).view(BS, BS - 1)  # (bs, bs-1)
    sim1_pos_neg = torch.cat((sim_pos1, sim_neg1_all), dim=1) / 0.07  # (bs, bs)
    loss1 = -F.log_softmax(sim1_pos_neg, dim=1)[:, 0].mean()

    # match fea2 to fea1
    sim_pos2 = (fea2 * fea1).sum(dim=1).unsqueeze(-1)  # (bs,1)
    sim_neg2_all = (fea2.unsqueeze(1) * fea1.unsqueeze(0)).sum(dim=-1)  # (bs,bs)
    sim_neg2_all = torch.masked_select(sim_neg2_all, unmask == 0).view(BS, BS - 1)  # (bs, bs-1)
    sim2_pos_neg = torch.cat((sim_pos2, sim_neg2_all), dim=1) / 0.07  # (bs, bs)
    loss2 = -F.log_softmax(sim2_pos_neg, dim=1)[:, 0].mean()

    loss = (loss1 + loss2) / 2.0
    return loss


def AlignTripLoss(fea1, fea2, mask1, mask2):
    # fea1_size (bs, max_len1, dim)  mask1_size (bs, max_len1)
    # fea2_size (bs, max_len2, dim)  mask2_size (bs, max_len2)
    # '-Inf' for padded item, '0' for others
    fea1 = F.normalize(fea1, p=2, dim=-1)
    fea2 = F.normalize(fea2, p=2, dim=-1)

    # match fea1 to fea2
    sim_pos1 = cal_sim(fea1, fea2, mask1, mask2)  # (bs)
    # (bs, 1, max_len1, dim)  (1, bs, max_len2, dim)
    sim_neg1_all = cal_sim_all(fea1.unsqueeze(1), fea2.unsqueeze(0), mask1, mask2)  # (bs,bs)
    unmask = torch.eye(sim_pos1.size(0), dtype=torch.float32, device=sim_pos1.device)
    unmask = torch.where(unmask == 1, torch.tensor([float('-Inf')], device=unmask.device), unmask)

    sim_neg1, _ = torch.max(sim_neg1_all + unmask, 1)
    loss1 = -sim_pos1 + sim_neg1 + 0.2
    loss1 = torch.maximum(loss1, torch.zeros_like(loss1)).mean()

    # match fea2 to fea1
    sim_pos2 = cal_sim(fea2, fea1, mask2, mask1)  # (bs)
    # (bs, 1, max_len1, dim)  (1, bs, max_len2, dim)
    sim_neg2_all = cal_sim_all(fea2.unsqueeze(1), fea1.unsqueeze(0), mask2, mask1)  # (bs,bs)
    sim_neg2, _ = torch.max(sim_neg2_all + unmask, 1)
    loss2 = -sim_pos2 + sim_neg2 + 0.2
    loss2 = torch.maximum(loss2, torch.zeros_like(loss2)).mean()

    loss = loss1 + loss2

    return loss


def cal_sim_all(fea1, fea2, mask1, mask2):
    # fea1_size (bs, 1, max_len1, dim)  mask1_size (bs, max_len1)
    # fea2_size (1, bs, max_len2, dim)  mask2_size (bs, max_len2)
    # '-Inf' for padded item, '0' for others
    max_len1 = fea1.size(2)
    max_len2 = fea2.size(2)
    bs = fea1.size(0)
    fea1_tmp = fea1.unsqueeze(3)  # (bs, 1, max_len1, 1, dim)
    fea2_tmp = fea2.unsqueeze(2)  # (1, bs, 1, max_len2, dim)
    fea_sim = (fea1_tmp * fea2_tmp).sum(dim=-1)  # (bs, bs, max_len1, max_len2)

    fea_sim = fea_sim + mask2.unsqueeze(dim=1)  # (bs, bs, max_len1, max_len2)
    idxs = torch.argmax(fea_sim, dim=-1).view(-1).unsqueeze(-1)  # (bs*bs*max_len1, 1)
    fea_sim = fea_sim.view(-1, max_len2)  # (bs*bs*max_len1, max_len2)
    select_sim = torch.gather(fea_sim, 1, idxs).view(bs, bs, max_len1)  # (bs, bs, max_len1)
    mask1_mult = torch.where(mask1 == 0, torch.tensor([1.0], device=mask1.device),
                             torch.tensor([0.0], device=mask1.device)).unsqueeze(1)  # (bs, 1, max_len1)
    select_sim = (select_sim * mask1_mult).sum(dim=-1) / mask1_mult.sum(dim=-1)  # (bs, bs)

    return select_sim


def cal_sim(fea1, fea2, mask1, mask2):
    # fea1_size (bs, max_len1, dim)  mask1_size (bs, max_len1)
    # fea2_size (bs, max_len2, dim)  mask2_size (bs, max_len2)
    # '-Inf' for padded item, '0' for others
    max_len1 = fea1.size(1)
    max_len2 = fea2.size(1)
    fea1_tmp = fea1.unsqueeze(2)  # (bs, max_len1, 1, dim)
    fea2_tmp = fea2.unsqueeze(1)  # (bs, 1, max_len2, dim)
    fea_sim = (fea1_tmp * fea2_tmp).sum(dim=-1)  # (bs, max_len1, max_len2)

    fea_sim = fea_sim + mask2.unsqueeze(dim=1)  # (bs, max_len1, max_len2)
    idxs = torch.argmax(fea_sim, dim=-1).view(-1).unsqueeze(-1)  # (bs*max_len1, 1)
    fea_sim = fea_sim.view(-1, max_len2)  # (bs*max_len1, max_len2)
    select_sim = torch.gather(fea_sim, 1, idxs).view(-1, max_len1)  # (bs, max_len1)
    mask1_mult = torch.where(mask1 == 0, 1, 0)
    select_sim = (select_sim * mask1_mult).sum(dim=-1) / mask1_mult.sum(dim=-1)  # (bs)

    return select_sim


def alignmentLoss(fea1, fea2, mask1, mask2):
    # fea1_size (bs, max_len1, dim)  mask1_size (bs, max_len1)
    # fea2_size (bs, max_len2, dim)  mask2_size (bs, max_len2)
    # '-Inf' for padded item, '0' for others

    fea1 = F.normalize(fea1, p=2, dim=-1)
    fea2 = F.normalize(fea2, p=2, dim=-1)
    loss1 = alignSingleLoss(fea1, fea2, mask1, mask2)
    loss2 = alignSingleLoss(fea2, fea1, mask2, mask1)
    loss = (loss1 + loss2) / 2.0
    return loss


def attAlignmentLoss(fea1, fea2, mask1, mask2, attFc):
    # fea1_size (bs, max_len1, dim)  mask1_size (bs, max_len1)
    # fea2_size (bs, max_len2, dim)  mask2_size (bs, max_len2)
    # '-Inf' for padded item, '0' for others

    fea1 = F.normalize(fea1, p=2, dim=-1)
    fea2 = F.normalize(fea2, p=2, dim=-1)
    fea1_tmp = fea1.unsqueeze(2)  # (bs, max_len1, 1, dim)
    fea2_tmp = fea2.unsqueeze(1)  # (bs, 1, max_len2, dim)
    fea_sim = fea1_tmp * fea2_tmp
    att_sim = attFc(fea_sim).squeeze(-1)  # (bs, max_len1, max_len2)
    fea_sim = fea_sim.sum(dim=-1)  # (bs, max_len1, max_len2)
    fea_sim = fea_sim * att_sim  # (bs, max_len1, max_len2)

    ###Simple as max_len1=49 
    loss = torch.masked_select(fea_sim, (mask2 == 0).unsqueeze(1))
    loss = 1.0 - loss.mean()

    return loss


def alignSingleLoss(fea1, fea2, mask1, mask2):
    # fea1_size (bs, max_len1, dim)  mask1_size (bs, max_len1)
    # fea2_size (bs, max_len2, dim)  mask2_size (bs, max_len2)
    # '-Inf' for padded item, '0' for others

    fea1_tmp = fea1.unsqueeze(2)  # (bs, max_len1, 1, dim)
    fea2_tmp = fea2.unsqueeze(1)  # (bs, 1, max_len2, dim)
    fea_sim = (fea1_tmp * fea2_tmp).sum(dim=-1)  # (bs, max_len1, max_len2)
    fea_sim = fea_sim + mask2.unsqueeze(dim=1)  # (bs, max_len1, max_len2)
    idxs = torch.argmax(fea_sim, dim=-1).view(-1).unsqueeze(-1)  # (bs*max_len1, 1)
    fea_sim = fea_sim.view(-1, fea_sim.size(-1))  # (bs*max_len1, max_len2)

    select_sim = torch.gather(fea_sim, 1, idxs).view(-1)  # (bs*max_len1)
    select_sim = torch.masked_select(select_sim, (mask1 == 0).view(-1))
    loss = 1.0 - torch.mean(select_sim)
    return loss


def getLanMask(seq_lens, max_len):
    # seq_lens (bs)
    mask = torch.ones((seq_lens.size(0), max_len))  # (bs, max_len)
    idxs = torch.arange(max_len).unsqueeze(dim=0)  # (1, max_len)
    seq_lens = seq_lens.unsqueeze(-1)  # (bs, 1)
    mask = torch.where(idxs < seq_lens, mask, torch.Tensor([0.0]))
    return mask
