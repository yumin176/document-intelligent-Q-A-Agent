from app.ingestion.cleaning.dedup import DedupCleaner
from app.ingestion.cleaning.media import MediaCleaner
from app.ingestion.cleaning.noise import NoiseCleaner
from app.ingestion.cleaning.normalize import NormalizeCleaner
from app.ingestion.cleaning.pipeline import CleanerPipeline
from app.ingestion.cleaning.whitespace import WhitespaceCleaner


def build_default_cleaner_pipeline() -> CleanerPipeline:
    return CleanerPipeline(
        [
            NormalizeCleaner(),
            MediaCleaner(),
            NoiseCleaner(),
            WhitespaceCleaner(),
            DedupCleaner(),
        ]
    )
