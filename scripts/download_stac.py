import pandas as pd
from pystac import Catalog, Item
import shapely.geometry

catalog_uri = (
    "https://radiantearth.blob.core.windows.net/mlhub/rapidai4eo/stac-v1.0/catalog.json"
)
catalog = Catalog.from_file(catalog_uri)

for collection in catalog.get_collections():
    print(collection.id)

def items_intersecting_geometry(collection, geometry):
    """Recursively find all STAC items intersecting a geometry.

    Our STAC structure (further detailed in the corpus documentation) has a hierarchy of
    Collections to speed up spatial queries. Recursively search these Collections, and
    return all STAC Items that intersect our geometry.
    """
    intersecting_items = []
    collection_bboxes = [
        shapely.geometry.box(*bounds) for bounds in collection.extent.spatial.bboxes
    ]
    if any([bbox.intersects(geometry) for bbox in collection_bboxes]):

        # Collect all matching items in this collection
        for item in collection.get_items():
            item_geometry = shapely.geometry.shape(item.geometry)
            if item_geometry.intersects(geometry):
                intersecting_items.append(item)

                example_pf_assets = item.get_assets()
                for i, (asset_name, asset) in enumerate(example_pf_assets.items()):
                    print(f"Asset: {asset_name}:")
                    print(asset.to_dict())
                    print("---")

        # Recursively search our nested collections for items
        for subcollection in collection.get_collections():
            intersecting_items += items_intersecting_geometry(subcollection, geometry)

    return intersecting_items


berlin_bbox = shapely.geometry.box(13.05, 52.35, 13.72, 52.69)
# berlin_bbox = shapely.geometry.box(2.096148, 41.497630, 2.115387, 41.505287)
pf_collection = catalog.get_child("rapidai4eo_v1_source_pf")

berlin_pf_items = items_intersecting_geometry(pf_collection, berlin_bbox)
print(f"Found {len(berlin_pf_items)} in the vicinity of Berlin.")

example_pf_item = berlin_pf_items[0]
example_pf_assets = example_pf_item.get_assets()
n_head = 5

for i, (asset_name, asset) in enumerate(example_pf_assets.items()):
    print(f"Asset: {asset_name}:")
    print(asset.to_dict())
    print("---")

    if i >= n_head - 1:
        break
