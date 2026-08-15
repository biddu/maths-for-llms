"""The 1 B-shaped model used by the Chapter 2 tying box.  Same vocabulary as
Model D, a quarter of the width, half the layers."""
from arith.model_d import Config, non_embedding, embedding

SMALL = Config(L=16, d=2048, h=16, d_h=128, n_kv=4, d_ff=8192, V=128256)


def report(c: Config = SMALL) -> None:
    n, e = non_embedding(c), embedding(c)
    print(f"non-embedding   {n:,}")
    print(f"V*d             {e:,}")
    print(f"untied total    {n + 2*e:,}   embedding share {100*2*e/(n+2*e):.1f}%")
    print(f"tied total      {n + e:,}")


if __name__ == "__main__":
    report()
