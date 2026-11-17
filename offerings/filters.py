import django_filters

from .models import Offering


class OfferingFilter(django_filters.FilterSet):
    min_seats = django_filters.NumberFilter(field_name="available_count", lookup_expr="gte")
    organization = django_filters.NumberFilter(field_name="organization_id")

    origin = django_filters.CharFilter(lookup_expr="icontains")
    destination = django_filters.CharFilter(lookup_expr="icontains")
    min_price = django_filters.NumberFilter(field_name="price", lookup_expr="gte")
    max_price = django_filters.NumberFilter(field_name="price", lookup_expr="lte")
    date_from = django_filters.DateFilter(field_name="starts_at__date", lookup_expr="gte")
    date_to = django_filters.DateFilter(field_name="starts_at__date", lookup_expr="lte")

    class Meta:
        model = Offering
        fields = ["category", "organization"]