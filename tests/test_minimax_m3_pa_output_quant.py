# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 sparse PA writing o_proj's per-token FP8 input (aiter #7401's
``pa_decode(quantized_output=, output_scale=, ...)``): the gate, the routing
and, on gfx1250, the numbers against FP32 attention."""

import inspect
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="base_attention defines @triton.jit kernels")
pytest.importorskip("aiter", reason="base_attention imports the AITER runtime")

from atom.model_ops import base_attention
from atom.model_ops.minimax_m3 import sparse_attn
from atom.models.minimax_m3 import model as m3_model

_QUANT_KWARGS = "quantized_output output_scale store_output quantized_pad_rows"


@pytest.mark.parametrize(
    "arch, kwargs, want",
    [
        ("gfx1250", _QUANT_KWARGS, True),
        ("gfx1250", "quantized_output output_scale", False),  # an older aiter
        ("gfx950", _QUANT_KWARGS, False),
    ],
)
def test_the_fp8_output_needs_aiter_to_take_all_four_kwargs(
    monkeypatch, arch, kwargs, want
):
    import importlib

    from aiter.jit.utils import chip_info

    def pa_decode():
        pass

    pa_decode.__signature__ = inspect.Signature(
        inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY)
        for n in ["output", *kwargs.split()]
    )
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: arch)
    monkeypatch.setattr(
        importlib, "import_module", lambda _: SimpleNamespace(pa_decode=pa_decode)
    )
    base_attention._flydsl_output_quant_api.cache_clear()
    try:
        assert base_attention._flydsl_output_quant_api() is want
    finally:
        base_attention._flydsl_output_quant_api.cache_clear()


class _Attention:
    def __init__(self):
        self.calls = []

    def forward_quantized(self, q, k, v, positions, qkv=None):
        self.calls.append("fp8")
        return q.to(torch.uint8), q.new_ones((q.shape[0], 1))

    def __call__(self, q, *args, **kwargs):
        self.calls.append("bf16")
        return q


@pytest.mark.parametrize(
    "supported, o_proj_fp8, want",
    [(True, True, "fp8"), (True, False, "bf16"), (False, True, "bf16")],
)
def test_o_proj_takes_the_pa_fp8_when_it_is_per_token_fp8_at_forward(
    supported, o_proj_fp8, want
):
    """o_proj's quantization is read at forward: online quantization re-quantizes
    it after construction."""
    seen = []

    def o_proj(x, x_scale=None):
        seen.append((x.dtype, x_scale))
        return x

    o_proj.quant_type = m3_model.QuantType.per_Token
    o_proj.params_dtype = m3_model.dtypes.fp8 if o_proj_fp8 else torch.bfloat16
    layer = SimpleNamespace(
        qkv_proj=lambda x, x_scale=None: x,
        o_proj=o_proj,
        attn=_Attention(),
        q_size=4,
        kv_size=2,
        index_q_size=2,
        idx_head_dim=2,
        sparse_pa_output_quant=supported,
    )
    m3_model.MiniMaxM3SparseAttention.forward(layer, torch.arange(3), torch.ones(3, 12))
    assert layer.attn.calls == [want]
    ((dtype, scale),) = seen
    if want == "fp8":
        assert dtype == torch.uint8 and scale.shape == (3, 1)
    else:
        assert dtype == torch.float32 and scale is None


def test_only_atom_sparse_mha_outside_sp_quantizes_its_output():
    from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl
    from atom.model_ops.paged_attention import Attention

    impl = object.__new__(SparseMHAPagedAttentionImpl)
    attn = object.__new__(Attention)
    torch.nn.Module.__init__(attn)
    attn.use_mla, attn.sp_columns, attn.impl = False, None, impl
    impl.kv_cache_dtype = "fp8"
    assert attn.quantizes_output
    impl.kv_cache_dtype = "fp4"  # the MXFP4 KV's PA has no FP8 output
    assert not attn.quantizes_output
    impl.kv_cache_dtype, attn.sp_columns = "fp8", object()  # Ulysses SP
    assert not attn.quantizes_output


@pytest.mark.parametrize(
    "decode, splits, rows, cap, budget",
    [
        (False, 4, 5, 1, None),  # prefill rows: one CTA each, the fused direct kernel
        (False, 8, 5, 8, None),  # eight splits: the fused cluster8 merge
        (True, 4, 128, 8, 8 * 128),  # decode rows: one resident cluster of eight
        (True, 8, 72, 8, 8 * 72),  # eight splits, but not the default's clusters
        (True, 4, 241, 4, None),  # past the resident clusters: the split stays
    ],
)
def test_sparse_rows_pick_a_split_the_fp8_epilogue_serves(
    monkeypatch, decode, splits, rows, cap, budget
):
    seen = {}
    monkeypatch.setattr(base_attention, "run_pa_decode", lambda **kw: seen.update(kw))
    monkeypatch.setattr(
        "aiter.ops.triton.gluon.pa_decode_gluon.get_recommended_splits",
        lambda *_: splits,
    )
    monkeypatch.setattr(sparse_attn, "_sparse_pa_ran_on_asm", pytest.fail)
    monkeypatch.setattr(sparse_attn, "_cluster_resident_rows", lambda _: 240)
    bucket = rows + 3  # a padded batch: the PA writes the last three inert
    q = torch.randn(rows, 16, 128, dtype=torch.bfloat16)
    fp8 = torch.empty((bucket, 16, 128), dtype=torch.uint8)
    scale = torch.empty((bucket, 1))
    sparse_attn.sparse_pa_per_row(
        *(q, torch.empty((1, 1, 8, 16, 16), dtype=torch.uint8)),
        torch.empty((1, 1, 1, 128, 16), dtype=torch.uint8),
        *(torch.zeros((rows, 8), dtype=torch.int32), torch.ones(rows)),
        *(1, 1.0, torch.empty_like(q), None, None),
        quantized_output=fp8,
        output_scale=scale,
        decode=decode,
    )
    assert seen["plan_partition_cap"] == seen["max_context_partition_num"] == cap
    assert seen.get("plan_workgroup_budget") == budget
    assert seen["q"].shape[0] == rows and seen["output_scale"] is scale
    assert seen["quantized_output"].shape == (bucket, 16, 128)
    assert seen["quantized_output"].data_ptr() == fp8.data_ptr()


def test_run_pa_decode_hands_flydsl_the_fp8_output_alone(monkeypatch):
    """No BF16 store, and the rows past q's written inert by the same launch."""
    import aiter.ops.flydsl.pa_decode as fly

    seen = {}
    ba = base_attention
    monkeypatch.setattr(ba, "flydsl_decode_seqs", lambda **kw: 5)
    monkeypatch.setattr(ba, "_flydsl_context_bound", lambda *a: 2048)
    monkeypatch.setattr(ba, "_flydsl_prepare_explicit_plan", lambda **kw: None)
    monkeypatch.setattr(ba, "_flydsl_plan_scratch", lambda *a, **kw: (None,) * 3)
    monkeypatch.setattr(fly, "pa_decode", lambda *a, **kw: seen.update(kw))

    def m(*shape):
        return torch.empty(shape, device="meta")

    def call(**kw):
        ba.run_pa_decode(
            *(m(5, 16, 128), m(5, 16, 128), m(8, 1, 8, 16, 16), m(8, 1, 1, 128, 16)),
            *(m(5), m(5, 8), 1.0, 1, 4, 256, torch.bfloat16, None, None, None),
            **kw,
        )

    fp8, scale = m(8, 16, 128), m(8, 1)
    call(quantized_output=fp8, output_scale=scale)
    assert seen["quantized_output"] is fp8 and seen["output_scale"] is scale
    assert seen["store_output"] is False and seen["quantized_pad_rows"] == 3
    with pytest.raises(ValueError, match="go together"):
        call(quantized_output=fp8)
    monkeypatch.setattr(ba, "flydsl_decode_seqs", lambda **kw: None)
    with pytest.raises(RuntimeError, match="only FlyDSL"):
        call(quantized_output=fp8, output_scale=scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "decode, rows", [(True, 16), (True, 72), (True, 128), (False, 300)]
)
def test_fp8_output_dequantizes_to_attention(monkeypatch, decode, rows):
    """FP8 rows times their descales against FP32 attention over the same FP8
    pages; two pad rows past q's come back zero with a unit descale."""
    from aiter import dtypes

    if not base_attention._flydsl_output_quant_api():
        pytest.skip("aiter's FlyDSL pa_decode has no FP8 output (ROCm/aiter#7401)")
    monkeypatch.setattr(base_attention.envs, "ATOM_PA_FLYDSL", True)
    torch.manual_seed(0)
    dev, pages, window, pad = "cuda", 4096, 2048, 2
    q = torch.randn(rows, 16, 128, device=dev, dtype=torch.bfloat16)
    # SHUFFLE pages of 16 tokens: K [page, 1, d // 16, token, 16], V [page, 1, 1, d, token]
    k_cache = (torch.randn(pages, 1, 8, 16, 16, device=dev) / 4).to(dtypes.fp8)
    v_cache = (torch.randn(pages, 1, 1, 128, 16, device=dev) / 4).to(dtypes.fp8)
    kv_scale = torch.ones(pages, 1, 16, device=dev)
    # A decode row attends a whole top-k window, a prefill row a causal 1..2048.
    lens = torch.randint(1, window + 1, (rows,), dtype=torch.int32, device=dev)
    if decode:
        lens.fill_(window)
    bt = torch.randint(0, pages, (rows, window // 16), dtype=torch.int32, device=dev)
    fp8 = torch.full((rows + pad, 16, 128), 0x7F, device=dev, dtype=torch.uint8)
    fp8 = fp8.view(dtypes.fp8)
    scale = torch.full((rows + pad, 1), float("nan"), device=dev)
    sparse_attn.sparse_pa_per_row(
        *(q, k_cache, v_cache, bt, lens, 1, 128**-0.5, torch.empty_like(q)),
        *(kv_scale, kv_scale),
        quantized_output=fp8,
        output_scale=scale,
        decode=decode,
    )

    keys = k_cache.float()[:, 0].permute(0, 2, 1, 3).flatten(2)[bt.long()]
    values = v_cache.float()[:, 0, 0].transpose(1, 2)[bt.long()]
    logits = torch.einsum("rhd,rtd->rht", q.float(), keys.flatten(1, 2)) / 128**0.5
    past = torch.arange(window, device=dev) >= lens[:, None, None]
    probs = logits.masked_fill(past, float("-inf")).softmax(-1)
    ref = torch.einsum("rht,rtd->rhd", probs, values.flatten(1, 2))
    got = fp8[:rows].float() * scale[:rows, :, None]
    amax = ref.abs().amax(dim=(1, 2), keepdim=True)
    assert ((got - ref).abs() <= amax / 10).all()  # + half an e4m3 step, amax / 28
    assert (fp8[rows:].float() == 0).all() and (scale[rows:] == 1).all()


def test_a_padded_token_graph_prefill_attends_its_scheduled_rows(monkeypatch):
    """The quant op hands the impl the scheduled rows and the bucket's whole FP8
    output, whose rows past them the PA writes inert (`quantized_pad_rows`)."""
    from atom.utils import forward_context

    seen = {}
    layer = SimpleNamespace(impl=SimpleNamespace(forward=lambda **kw: seen.update(kw)))
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"attn": layer})
    )
    monkeypatch.setattr(base_attention, "get_current_atom_config", lambda: config)
    step = SimpleNamespace(is_prefill=True, scheduled_tokens=5)
    monkeypatch.setattr(
        forward_context, "_forward_context", SimpleNamespace(context=step)
    )
    q, fp8, scale = torch.ones(8, 4), torch.empty(8, 4), torch.empty(8, 1)
    base_attention.unified_attention_with_output_quant_base.__wrapped__(
        q, q, q, torch.arange(8), "attn", q, fp8, scale
    )
    assert [seen[n].shape[0] for n in ("query", "key", "value", "qkv")] == [5] * 4
    assert seen["position"].shape == (5,)
    assert seen["quantized_output"] is fp8 and seen["output_scale"] is scale
